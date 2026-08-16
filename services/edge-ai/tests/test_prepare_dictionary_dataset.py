import csv
import io
import json
from zipfile import ZIP_DEFLATED, ZipFile

from src.prepare_dictionary_dataset import records_from_archive, write_manifest


def test_creates_vocabulary_and_manifest(tmp_path) -> None:
    archive_path = tmp_path / 'dataset.zip'
    with ZipFile(archive_path, 'w', ZIP_DEFLATED) as archive:
        content = io.StringIO()
        writer = csv.DictWriter(content, fieldnames=['ID', 'VIDEO', 'LABEL'])
        writer.writeheader()
        writer.writerows(({'ID': 1, 'VIDEO': 'A.mp4', 'LABEL': 'a'}, {'ID': 2, 'VIDEO': 'B.mp4', 'LABEL': 'a'}, {'ID': 3, 'VIDEO': 'C.mp4', 'LABEL': 'b'}))
        archive.writestr('Dataset/Labels/label.csv', content.getvalue())
        for name in ('A.mp4', 'B.mp4', 'C.mp4'):
            archive.writestr(f'Dataset/Videos/{name}', b'video')
    with ZipFile(archive_path) as archive:
        records, vocabulary = records_from_archive(archive)
    output = tmp_path / 'manifest'
    write_manifest(records, vocabulary, output)
    assert vocabulary == ['a', 'b']
    assert records[2]['class_id'] == 1
    assert json.loads((output / 'labels.json').read_text(encoding='utf-8'))['class_count'] == 2
