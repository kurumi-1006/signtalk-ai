"""Evaluate a labelled local video dataset through the running Edge AI service.

The output is an append-only JSONL review file: each record preserves the
dataset's human label as well as the model's top predictions.  It can therefore
be used for a full-dataset audit without ever treating a pseudo-label as ground
truth or silently changing the model's vocabulary.
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import ZipFile

import httpx

VIDEO_SUFFIXES = {'.avi', '.m4v', '.mov', '.mp4', '.webm'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Create reviewable pseudo-labels from a video dataset.')
    parser.add_argument('--input', type=Path, required=True, help='Video directory or a .zip dataset archive.')
    parser.add_argument('--output', type=Path, required=True, help='Destination JSONL file.')
    parser.add_argument('--edge-url', default='http://localhost:8082', help='Running Edge AI base URL.')
    parser.add_argument('--device-id', default='uno-q-demo')
    parser.add_argument('--min-confidence', type=float, default=0.82)
    parser.add_argument('--min-margin', type=float, default=0.20)
    parser.add_argument('--min-landmark-coverage', type=float, default=0.50)
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--limit', type=int, help='Only process this many videos (useful for a smoke test).')
    parser.add_argument('--resume', action='store_true', help='Skip videos already present in OUTPUT and append new records.')
    parser.add_argument('--shard-index', type=int, default=0, help='Zero-based shard to run (requires --shard-count).')
    parser.add_argument('--shard-count', type=int, default=1, help='Number of deterministic, non-overlapping shards.')
    return parser.parse_args()


def zip_members(archive: ZipFile) -> Iterable[str]:
    for member in archive.namelist():
        path = PurePosixPath(member)
        if path.suffix.lower() in VIDEO_SUFFIXES and not path.is_absolute() and '..' not in path.parts:
            yield member


def labels_from_rows(rows: Iterable[dict[str, str]]) -> dict[str, str]:
    """Read either SignTalk's ground_truth.csv or VSL Dictionary's label.csv."""
    labels: dict[str, str] = {}
    for row in rows:
        video = row.get('file_name') or row.get('VIDEO')
        label = row.get('ground_truth') or row.get('LABEL')
        if video and label:
            labels[Path(video).name] = label
    return labels


def ground_truth_from_zip(archive: ZipFile) -> dict[str, str]:
    members = set(archive.namelist())
    if 'ground_truth.csv' in members:
        content = archive.read('ground_truth.csv').decode('utf-8-sig').splitlines()
        return labels_from_rows(csv.DictReader(content))
    label_csv = next((member for member in members if member.casefold().endswith('/labels/label.csv')), None)
    if label_csv is None:
        return {}
    content = archive.read(label_csv).decode('utf-8-sig').splitlines()
    return labels_from_rows(csv.DictReader(content))


def processed_videos(output: Path) -> set[str]:
    if not output.is_file():
        return set()
    completed: set[str] = set()
    with output.open(encoding='utf-8') as stream:
        for line in stream:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and isinstance(value.get('video'), str):
                completed.add(value['video'])
    return completed


def append_record(output: Path, record: dict[str, Any]) -> None:
    """Persist each completed inference so a long evaluation can be resumed."""
    with output.open('a', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + '\n')


def predict(client: httpx.Client, edge_url: str, device_id: str, path: Path) -> dict[str, Any]:
    with path.open('rb') as video:
        response = client.post(
            f'{edge_url.rstrip("/")}/predict',
            data={'device_id': device_id},
            files={'video': (path.name, video, 'video/mp4')},
        )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get('event'), dict):
        raise TypeError('Edge AI response does not contain an event.')
    return payload


def make_record(
    source: str,
    payload: dict[str, Any],
    thresholds: tuple[float, float, float],
    dataset_label: str | None = None,
) -> dict[str, Any]:
    event = payload['event']
    prediction = event['payload']
    confidence = float(prediction['confidence'])
    margin = float(prediction.get('margin', 0))
    coverage = float(prediction.get('landmarkCoverage', 0))
    accepted = confidence >= thresholds[0] and margin >= thresholds[1] and coverage >= thresholds[2]
    return {
        'video': source,
        'dataset_label': dataset_label,
        # A textual equality comparison is deliberately omitted here.  The
        # Multi-VSL checkpoint emits its own 1,000-class indices, whereas an
        # arbitrary VSL Dictionary export may contain a different vocabulary.
        # This field makes that distinction explicit for downstream review.
        'dataset_label_comparison': 'requires_model_class_map' if dataset_label else 'not_available',
        'predicted_label': prediction['label'],
        'confidence': confidence,
        'margin': margin,
        'landmark_coverage': coverage,
        'top_k': prediction.get('topK', [])[:3],
        'model_id': payload.get('diagnostics', {}).get('model_id'),
        'pseudo_label_status': 'accepted_for_review' if accepted else 'needs_review',
        'review_status': 'pending',
        'eligible_for_training': False,
    }


def process_path(
    source: str,
    path: Path,
    dataset_label: str | None,
    client: httpx.Client,
    args: argparse.Namespace,
) -> dict[str, Any]:
    try:
        payload = predict(client, args.edge_url, args.device_id, path)
        return make_record(
            source,
            payload,
            (args.min_confidence, args.min_margin, args.min_landmark_coverage),
            dataset_label,
        )
    except (httpx.HTTPError, OSError, ValueError) as error:
        return {
            'video': source,
            'dataset_label': dataset_label,
            'pseudo_label_status': 'failed',
            'review_status': 'pending',
            'eligible_for_training': False,
            'error': str(error),
        }


def main() -> None:
    args = parse_args()
    if not 0 <= args.min_confidence <= 1 or not 0 <= args.min_margin <= 1 or not 0 <= args.min_landmark_coverage <= 1:
        raise SystemExit('All thresholds must be between 0 and 1.')
    if not args.input.exists():
        raise SystemExit(f'Input does not exist: {args.input}')

    if args.limit is not None and args.limit <= 0:
        raise SystemExit('--limit must be positive.')
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise SystemExit('--shard-count must be positive and --shard-index must be in 0..shard-count-1.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.resume:
        raise SystemExit(f'Output already exists: {args.output}. Use --resume or choose a new path.')
    completed = processed_videos(args.output) if args.resume else set()
    records: list[dict[str, Any]] = []
    with httpx.Client(timeout=args.timeout) as client:
        if args.input.suffix.lower() == '.zip':
            with ZipFile(args.input) as archive, tempfile.TemporaryDirectory(prefix='signtalk-pseudolabel-') as temporary:
                root = Path(temporary)
                truth = ground_truth_from_zip(archive)
                for index, member in enumerate(sorted(zip_members(archive))):
                    if index % args.shard_count != args.shard_index:
                        continue
                    if member in completed:
                        continue
                    if args.limit is not None and len(records) >= args.limit:
                        break
                    path = root / PurePosixPath(member)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(archive.read(member))
                    record = process_path(member, path, truth.get(path.name), client, args)
                    append_record(args.output, record)
                    records.append(record)
        else:
            for index, path in enumerate(sorted(item for item in args.input.rglob('*') if item.suffix.lower() in VIDEO_SUFFIXES)):
                if index % args.shard_count != args.shard_index:
                    continue
                source = str(path.relative_to(args.input))
                if source in completed:
                    continue
                if args.limit is not None and len(records) >= args.limit:
                    break
                record = process_path(source, path, None, client, args)
                append_record(args.output, record)
                records.append(record)
    summary = {
        'total': len(records),
        'accepted_for_review': sum(item.get('pseudo_label_status') == 'accepted_for_review' for item in records),
        'needs_review': sum(item.get('pseudo_label_status') == 'needs_review' for item in records),
        'failed': sum(item.get('pseudo_label_status') == 'failed' for item in records),
        'output': str(args.output),
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
