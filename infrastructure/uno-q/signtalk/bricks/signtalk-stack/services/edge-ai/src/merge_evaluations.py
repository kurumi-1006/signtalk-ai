"""Merge independent/resumable evaluation shards into one JSONL report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Merge SignTalk evaluation JSONL shards by video path.')
    parser.add_argument('--input', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def merge(inputs: list[Path]) -> list[dict[str, Any]]:
    by_video: dict[str, dict[str, Any]] = {}
    for path in inputs:
        with path.open(encoding='utf-8') as stream:
            for number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get('video'), str):
                    raise TypeError(f'Invalid record at {path}:{number}')
                by_video[row['video']] = row
    return [by_video[video] for video in sorted(by_video)]


def main() -> None:
    args = parse_args()
    missing = [str(path) for path in args.input if not path.is_file()]
    if missing:
        raise SystemExit(f'Input does not exist: {", ".join(missing)}')
    records = merge(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8', newline='\n') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(json.dumps({'videos': len(records), 'output': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
