"""Wait for all audit shards, then install clearly marked guessed class labels."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import httpx
import torch

from .derive_guessed_class_map import add_nearest_head_guesses, aggregate, build_candidates, guessed_labels
from .merge_evaluations import merge
from .summarize_evaluation import write_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Finalize a sharded VSL Dictionary audit.')
    parser.add_argument('--input', type=Path, nargs='+', required=True)
    parser.add_argument('--merged-output', type=Path, required=True)
    parser.add_argument('--labels-output', type=Path, required=True)
    parser.add_argument('--candidates-output', type=Path, required=True)
    parser.add_argument('--review-output', type=Path, required=True)
    parser.add_argument('--summary-output', type=Path, required=True)
    parser.add_argument('--expected-videos', type=int, required=True)
    parser.add_argument('--edge-url', default='http://127.0.0.1:8082')
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--checkpoint', type=Path, help='Optional checkpoint for no-evidence nearest-head guesses.')
    return parser.parse_args()


def finalize(args: argparse.Namespace) -> int:
    if not all(path.is_file() for path in args.input):
        return 0
    records = merge(args.input)
    if len(records) < args.expected_videos:
        return len(records)
    if len(records) != args.expected_videos:
        raise ValueError(f'Expected {args.expected_videos} unique videos, got {len(records)}.')
    args.merged_output.parent.mkdir(parents=True, exist_ok=True)
    with args.merged_output.open('w', encoding='utf-8', newline='\n') as stream:
        import json
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    write_outputs(records, args.review_output, args.summary_output)
    candidates = build_candidates(aggregate(args.merged_output, 1000), 1000)
    # The user explicitly asked for guesses. Every observed class therefore
    # receives its best dataset-based guess; unobserved IDs retain raw output.
    labels = guessed_labels(candidates, min_videos=1, min_dominance=0.0)
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        weights = checkpoint.get('head.1.weight') if isinstance(checkpoint, dict) else None
        if not isinstance(weights, torch.Tensor):
            raise ValueError(f'Checkpoint has no classifier head: {args.checkpoint}')
        labels = add_nearest_head_guesses(labels, candidates, weights)
    import json
    args.labels_output.parent.mkdir(parents=True, exist_ok=True)
    args.candidates_output.parent.mkdir(parents=True, exist_ok=True)
    args.labels_output.write_text(json.dumps({'id_to_label': labels, 'mapping_kind': 'guessed_from_full_dataset_audit'}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    args.candidates_output.write_text(json.dumps(candidates, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    response = httpx.post(f'{args.edge_url.rstrip("/")}/models/multi_vsl_mvit_v2_1000/activate', timeout=180)
    response.raise_for_status()
    return len(records)


def main() -> None:
    args = parse_args()
    if args.expected_videos <= 0 or args.poll_seconds <= 0:
        raise SystemExit('--expected-videos and --poll-seconds must be positive.')
    while True:
        completed = finalize(args)
        if completed == args.expected_videos:
            print(f'Installed guessed labels after auditing {completed} videos.')
            return
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
