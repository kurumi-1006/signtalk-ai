"""Turn a resumable JSONL model audit into an Excel-friendly review sheet."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Summarize a SignTalk JSONL dataset evaluation.')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--csv', type=Path, required=True, dest='csv_output')
    parser.add_argument('--summary', type=Path, required=True)
    return parser.parse_args()


def read_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f'Record {number} in {path} is not an object.')
            records.append(value)
    return records


def review_row(record: dict[str, Any]) -> dict[str, object]:
    candidates = record.get('top_k')
    top_k = candidates if isinstance(candidates, list) else []
    values: dict[str, object] = {
        'video': record.get('video', ''),
        'dataset_label': record.get('dataset_label', ''),
        'predicted_label': record.get('predicted_label', ''),
        'confidence': record.get('confidence', ''),
        'margin': record.get('margin', ''),
        'status': record.get('pseudo_label_status', ''),
        'error': record.get('error', ''),
    }
    for position in range(3):
        candidate = top_k[position] if position < len(top_k) and isinstance(top_k[position], dict) else {}
        values[f'top_{position + 1}_label'] = candidate.get('label', '')
        values[f'top_{position + 1}_confidence'] = candidate.get('confidence', '')
    return values


def summarize(records: list[dict[str, Any]]) -> dict[str, object]:
    statuses = Counter(str(record.get('pseudo_label_status', 'unknown')) for record in records)
    labels = {record.get('dataset_label') for record in records if isinstance(record.get('dataset_label'), str)}
    grouped_predictions: dict[str, list[str]] = {}
    for record in records:
        dataset_label = record.get('dataset_label')
        predicted_label = record.get('predicted_label')
        if isinstance(dataset_label, str) and isinstance(predicted_label, str):
            grouped_predictions.setdefault(dataset_label, []).append(predicted_label)
    repeated = [predictions for predictions in grouped_predictions.values() if len(predictions) >= 2]
    consistent = sum(len(set(predictions)) == 1 for predictions in repeated)
    return {
        'evaluated_videos': len(records),
        'distinct_dataset_labels_seen': len(labels),
        'status_counts': dict(sorted(statuses.items())),
        'requires_model_class_map': any(record.get('dataset_label_comparison') == 'requires_model_class_map' for record in records),
        # This is a valid quality signal before a semantic class map exists:
        # different views/examples of the same human label should converge on
        # one checkpoint class.  It is intentionally not called accuracy.
        'labels_with_multiple_videos': len(repeated),
        'top1_consistent_labels': consistent,
        'top1_label_consistency_rate': round(consistent / len(repeated), 6) if repeated else None,
    }


def write_outputs(records: list[dict[str, Any]], csv_output: Path, summary_output: Path) -> None:
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    columns = list(review_row({}).keys())
    with csv_output.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(review_row(record) for record in records)
    summary_output.write_text(json.dumps(summarize(records), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f'Input does not exist: {args.input}')
    records = read_records(args.input)
    write_outputs(records, args.csv_output, args.summary)
    print(json.dumps(summarize(records), ensure_ascii=False))


if __name__ == '__main__':
    main()
