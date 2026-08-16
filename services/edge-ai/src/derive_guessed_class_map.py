"""Infer transparent *guessed* names for MViT class IDs from an audit JSONL.

The supplied checkpoint has no semantic 1,000-class vocabulary.  This utility
does not claim to recover its training labels: it votes across labelled videos
and writes every derived name as a clearly marked guess, retaining the original
``vsl_gloss_####`` ID in the display text.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

CLASS_ID = re.compile(r'^vsl_gloss_(\d{4})$')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Derive guessed display labels from per-video model audit output.')
    parser.add_argument('--input', type=Path, required=True, help='Merged evaluation JSONL.')
    parser.add_argument('--labels-output', type=Path, required=True, help='Partial labels.json used by the predictor.')
    parser.add_argument('--candidates-output', type=Path, required=True, help='Full evidence report for human review.')
    parser.add_argument('--class-count', type=int, default=1000)
    parser.add_argument('--min-videos', type=int, default=2, help='Minimum number of supporting labelled videos.')
    parser.add_argument('--min-dominance', type=float, default=0.4, help='Minimum share of an ID’s evidence owned by the leading label.')
    parser.add_argument('--include-unobserved', action='store_true', help='Also emit an explicit no-evidence display label for IDs never observed.')
    parser.add_argument('--checkpoint', type=Path, help='MViT checkpoint used to guess never-observed IDs from their nearest head.')
    return parser.parse_args()


def class_index(value: object) -> int | None:
    if not isinstance(value, str):
        return None
    match = CLASS_ID.fullmatch(value)
    return int(match.group(1)) if match else None


def aggregate(path: Path, class_count: int) -> dict[int, dict[str, dict[str, float]]]:
    scores: dict[int, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: {'score': 0.0, 'videos': 0.0, 'top1': 0.0}))
    with path.open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record: Any = json.loads(line)
            if not isinstance(record, dict):
                raise TypeError(f'Record {line_number} is not an object.')
            truth = record.get('dataset_label')
            candidates = record.get('top_k')
            if not isinstance(truth, str) or not isinstance(candidates, list):
                continue
            seen: set[int] = set()
            for position, candidate in enumerate(candidates):
                if not isinstance(candidate, dict):
                    continue
                index = class_index(candidate.get('label'))
                confidence = candidate.get('confidence')
                if index is None or not 0 <= index < class_count or not isinstance(confidence, (int, float)):
                    continue
                values = scores[index][truth]
                values['score'] += float(confidence)
                values['top1'] += float(position == 0)
                if index not in seen:
                    values['videos'] += 1
                    seen.add(index)
    return scores


def build_candidates(scores: dict[int, dict[str, dict[str, float]]], class_count: int) -> dict[str, list[dict[str, object]]]:
    output: dict[str, list[dict[str, object]]] = {}
    for index in range(class_count):
        choices = scores.get(index, {})
        total = sum(value['score'] for value in choices.values())
        output[str(index)] = [
            {
                'label': label,
                'evidence_score': round(values['score'], 6),
                'supporting_videos': int(values['videos']),
                'top1_videos': int(values['top1']),
                'dominance': round(values['score'] / total, 6) if total else 0.0,
            }
            for label, values in sorted(choices.items(), key=lambda item: item[1]['score'], reverse=True)
        ]
    return output


def guessed_labels(
    candidates: dict[str, list[dict[str, object]]], min_videos: int, min_dominance: float, include_unobserved: bool = False,
) -> dict[str, str]:
    labels: dict[str, str] = {}
    for raw_index, choices in candidates.items():
        if not choices:
            if include_unobserved:
                labels[raw_index] = f'chưa có bằng chứng (vsl_gloss_{int(raw_index):04d})'
            continue
        winner = choices[0]
        if int(winner['supporting_videos']) < min_videos or float(winner['dominance']) < min_dominance:
            continue
        labels[raw_index] = f"{winner['label']} (ước đoán · vsl_gloss_{int(raw_index):04d})"
    return labels


def add_nearest_head_guesses(
    labels: dict[str, str], candidates: dict[str, list[dict[str, object]]], head_weights: torch.Tensor,
) -> dict[str, str]:
    """Use the nearest classifier-head neighbour only for IDs with no video evidence."""
    if head_weights.ndim != 2 or head_weights.shape[0] != len(candidates):
        raise ValueError('Classifier head shape does not match candidate class count.')
    observed = [int(raw_index) for raw_index, choices in candidates.items() if choices]
    if not observed:
        return labels
    normalized = torch.nn.functional.normalize(head_weights.float(), dim=1)
    observed_weights = normalized[observed]
    for raw_index, choices in candidates.items():
        if choices:
            continue
        index = int(raw_index)
        neighbour = observed[int(torch.argmax(normalized[index] @ observed_weights.T).item())]
        value = candidates[str(neighbour)][0].get('label')
        if isinstance(value, str):
            labels[raw_index] = f'{value} (ước đoán lân cận · vsl_gloss_{index:04d})'
    return labels


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f'Input does not exist: {args.input}')
    if args.class_count <= 0 or args.min_videos <= 0 or not 0 <= args.min_dominance <= 1:
        raise SystemExit('Invalid class count, minimum-video count, or dominance threshold.')
    candidates = build_candidates(aggregate(args.input, args.class_count), args.class_count)
    labels = guessed_labels(candidates, args.min_videos, args.min_dominance, args.include_unobserved)
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        weights = checkpoint.get('head.1.weight') if isinstance(checkpoint, dict) else None
        if not isinstance(weights, torch.Tensor):
            raise SystemExit(f'Checkpoint has no classifier head: {args.checkpoint}')
        labels = add_nearest_head_guesses(labels, candidates, weights)
    args.labels_output.parent.mkdir(parents=True, exist_ok=True)
    args.candidates_output.parent.mkdir(parents=True, exist_ok=True)
    args.labels_output.write_text(json.dumps({'id_to_label': labels, 'mapping_kind': 'guessed_from_dataset_audit'}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    args.candidates_output.write_text(json.dumps(candidates, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'mapped_classes': len(labels), 'classes_with_evidence': sum(bool(item) for item in candidates.values())}, ensure_ascii=False))


if __name__ == '__main__':
    main()
