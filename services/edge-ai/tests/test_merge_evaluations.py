import json

from src.merge_evaluations import merge


def test_merges_by_video_and_keeps_latest_record(tmp_path) -> None:
    first = tmp_path / 'first.jsonl'
    second = tmp_path / 'second.jsonl'
    first.write_text(json.dumps({'video': 'b.mp4', 'value': 1}) + '\n', encoding='utf-8')
    second.write_text('\n'.join((json.dumps({'video': 'a.mp4', 'value': 2}), json.dumps({'video': 'b.mp4', 'value': 3}))) + '\n', encoding='utf-8')
    assert merge([first, second]) == [{'video': 'a.mp4', 'value': 2}, {'video': 'b.mp4', 'value': 3}]
