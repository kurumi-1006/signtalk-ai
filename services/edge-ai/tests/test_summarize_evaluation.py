import csv
import json

from src.summarize_evaluation import read_records, summarize, write_outputs


def test_writes_excel_friendly_review_and_summary(tmp_path) -> None:
    input_path = tmp_path / 'audit.jsonl'
    input_path.write_text(json.dumps({
        'video': 'Dataset/Videos/D0001.mp4', 'dataset_label': 'địa chỉ',
        'predicted_label': 'vsl_gloss_0001', 'confidence': 0.9,
        'margin': 0.8, 'pseudo_label_status': 'accepted_for_review',
        'dataset_label_comparison': 'requires_model_class_map',
        'top_k': [{'label': 'vsl_gloss_0001', 'confidence': 0.9}],
    }, ensure_ascii=False) + '\n', encoding='utf-8')
    records = read_records(input_path)
    output_csv = tmp_path / 'review.csv'
    output_summary = tmp_path / 'summary.json'
    write_outputs(records, output_csv, output_summary)
    with output_csv.open(encoding='utf-8-sig', newline='') as stream:
        assert next(csv.DictReader(stream))['dataset_label'] == 'địa chỉ'
    assert json.loads(output_summary.read_text(encoding='utf-8')) == summarize(records)
