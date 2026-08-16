import json

import pytest

from src.inference.class_labels import load_optional_labels


def test_missing_map_uses_stable_model_ids(tmp_path) -> None:
    assert load_optional_labels(str(tmp_path / 'missing.json'), 3, 'vsl_gloss_') == {
        0: 'vsl_gloss_0000', 1: 'vsl_gloss_0001', 2: 'vsl_gloss_0002',
    }


def test_partial_map_keeps_unreviewed_classes_as_ids(tmp_path) -> None:
    path = tmp_path / 'labels.json'
    path.write_text(json.dumps({'id_to_label': {'1': 'địa chỉ'}}), encoding='utf-8')
    assert load_optional_labels(str(path), 3, 'vsl_gloss_') == {
        0: 'vsl_gloss_0000', 1: 'địa chỉ', 2: 'vsl_gloss_0002',
    }


def test_out_of_range_map_is_rejected(tmp_path) -> None:
    path = tmp_path / 'labels.json'
    path.write_text(json.dumps({'3': 'invalid'}), encoding='utf-8')
    with pytest.raises(ValueError, match='outside'):
        load_optional_labels(str(path), 3, 'vsl_gloss_')
