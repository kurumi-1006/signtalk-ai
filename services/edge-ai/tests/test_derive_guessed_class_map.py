import json

import torch

from src.derive_guessed_class_map import (
    add_nearest_head_guesses,
    aggregate,
    build_candidates,
    guessed_labels,
)


def test_votes_for_a_transparent_guessed_label(tmp_path) -> None:
    report = tmp_path / 'audit.jsonl'
    rows = [
        {'dataset_label': 'địa chỉ', 'top_k': [{'label': 'vsl_gloss_0001', 'confidence': 0.8}]},
        {'dataset_label': 'địa chỉ', 'top_k': [{'label': 'vsl_gloss_0001', 'confidence': 0.7}]},
        {'dataset_label': 'tỉnh', 'top_k': [{'label': 'vsl_gloss_0001', 'confidence': 0.2}]},
    ]
    report.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
    candidates = build_candidates(aggregate(report, 2), 2)
    labels = guessed_labels(candidates, min_videos=2, min_dominance=0.4)
    assert labels == {'1': 'địa chỉ (ước đoán · vsl_gloss_0001)'}
    assert candidates['1'][0]['dominance'] == 0.882353


def test_unobserved_class_gets_an_explicit_no_evidence_label() -> None:
    assert guessed_labels({'0': [], '1': []}, min_videos=1, min_dominance=0, include_unobserved=True) == {
        '0': 'chưa có bằng chứng (vsl_gloss_0000)',
        '1': 'chưa có bằng chứng (vsl_gloss_0001)',
    }


def test_unobserved_class_uses_nearest_classifier_head_guess() -> None:
    candidates = {
        '0': [{'label': 'address'}],
        '1': [],
        '2': [{'label': 'province'}],
    }
    weights = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    labels = add_nearest_head_guesses({'0': 'address', '2': 'province'}, candidates, weights)
    assert labels['1'].endswith('vsl_gloss_0001)')
    assert labels['1'].startswith('address')
