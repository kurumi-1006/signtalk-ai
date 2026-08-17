"""Validated class-index to display-label mappings for classifier adapters."""

from __future__ import annotations

import json
from pathlib import Path


def load_optional_labels(path: str, class_count: int, fallback_prefix: str) -> dict[int, str]:
    """Load a partial or complete label map without inventing class meanings.

    A map may be a JSON list or an object containing ``id_to_label``.  Missing
    entries deliberately keep the model's stable fallback label.  This makes it
    safe to deploy a reviewed mapping incrementally rather than pretending that
    the row order in an unrelated dataset is the checkpoint's class order.
    """
    labels_path = Path(path)
    fallback = {index: f'{fallback_prefix}{index:04d}' for index in range(class_count)}
    if not path or not labels_path.is_file():
        return fallback
    payload = json.loads(labels_path.read_text(encoding='utf-8'))
    values = payload.get('id_to_label', payload) if isinstance(payload, dict) else payload
    if isinstance(values, list):
        values = {str(index): value for index, value in enumerate(values)}
    if not isinstance(values, dict):
        raise TypeError(f'Class label map must be a list or object: {labels_path}')
    for raw_index, raw_label in values.items():
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as error:
            raise ValueError(f'Invalid class index {raw_index!r} in {labels_path}') from error
        if not 0 <= index < class_count:
            raise ValueError(f'Class index {index} is outside 0..{class_count - 1} in {labels_path}')
        if not isinstance(raw_label, str) or not raw_label.strip():
            raise ValueError(f'Class {index} has an empty display label in {labels_path}')
        fallback[index] = raw_label.strip()
    return fallback
