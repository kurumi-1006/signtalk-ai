"""Build a reproducible 3,315-class training manifest from a VSL Dictionary ZIP."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Create a vocabulary and sample manifest for VSL Dictionary fine-tuning.')
    parser.add_argument('--input', type=Path, required=True, help='ZIP containing Dataset/Labels/label.csv and Dataset/Videos/.')
    parser.add_argument('--output-dir', type=Path, required=True)
    return parser.parse_args()


def label_member(members: set[str]) -> str:
    member = next((name for name in members if name.casefold().endswith('/labels/label.csv')), None)
    if member is None:
        raise ValueError('Archive is missing Dataset/Labels/label.csv.')
    return member


def records_from_archive(archive: ZipFile) -> tuple[list[dict[str, object]], list[str]]:
    members = set(archive.namelist())
    video_members = {PurePosixPath(name).name: name for name in members}
    labels = archive.read(label_member(members)).decode('utf-8-sig').splitlines()
    vocabulary: dict[str, int] = {}
    records: list[dict[str, object]] = []
    for row_number, row in enumerate(csv.DictReader(labels), start=2):
        video, label = row.get('VIDEO'), row.get('LABEL')
        if not video or not label:
            raise ValueError(f'Incomplete label row {row_number}.')
        member = video_members.get(video)
        if member is None:
            raise ValueError(f'Labelled video {video!r} is missing from the archive.')
        class_id = vocabulary.setdefault(label, len(vocabulary))
        records.append({'archive_member': member, 'video': video, 'class_id': class_id, 'label': label})
    return records, list(vocabulary)


def write_manifest(records: list[dict[str, object]], vocabulary: list[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels.json').write_text(
        json.dumps({'id_to_label': vocabulary, 'class_count': len(vocabulary)}, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    with (output_dir / 'samples.jsonl').open('w', encoding='utf-8', newline='\n') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    (output_dir / 'summary.json').write_text(
        json.dumps({'videos': len(records), 'classes': len(vocabulary)}, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f'Input does not exist: {args.input}')
    with ZipFile(args.input) as archive:
        records, vocabulary = records_from_archive(archive)
    write_manifest(records, vocabulary, args.output_dir)
    print(json.dumps({'videos': len(records), 'classes': len(vocabulary), 'output_dir': str(args.output_dir)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
