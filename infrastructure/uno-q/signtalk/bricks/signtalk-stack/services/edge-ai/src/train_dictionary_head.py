"""Fine-tune the Multi-VSL classifier head on a labelled VSL Dictionary ZIP.

This deliberately freezes the MViT backbone.  The supplied dictionary has many
one-shot classes, so this is a controlled adaptation baseline, not a claim of
held-out generalization.  Evaluate a signer-disjoint dataset before deployment.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .inference.multivsl.mvit_v2 import mvit_v2_s
from .prepare_dictionary_dataset import records_from_archive


class DictionaryDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, archive_path: Path, records: list[dict[str, object]]) -> None:
        self.archive_path = archive_path
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def tensor_from_frames(frames: list[np.ndarray]) -> torch.Tensor:
        if not frames:
            raise ValueError('Video has no decodable frames.')
        indices = np.linspace(0, len(frames) - 1, 16).round().astype(np.int64)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        output: list[np.ndarray] = []
        for index in indices:
            frame = frames[int(index)]
            rgb = cv2.cvtColor(cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            output.append((rgb.astype(np.float32) / 255.0 - mean) / std)
        return torch.from_numpy(np.stack(output).transpose(3, 0, 1, 2))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        member = record['archive_member']
        class_id = record['class_id']
        if not isinstance(member, str) or not isinstance(class_id, int):
            raise TypeError('Invalid dictionary manifest record.')
        with ZipFile(self.archive_path) as archive, tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(archive.read(member))
        try:
            capture = cv2.VideoCapture(str(temporary_path))
            frames: list[np.ndarray] = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame)
            capture.release()
        finally:
            temporary_path.unlink(missing_ok=True)
        return self.tensor_from_frames(frames), torch.tensor(class_id, dtype=torch.long)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Fine-tune a Multi-VSL MViT head for the labelled VSL Dictionary.')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--base-checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    return parser.parse_args()


def load_model(checkpoint: Path, class_count: int) -> nn.Module:
    state_dict = torch.load(checkpoint, map_location='cpu')
    model = mvit_v2_s(num_classes=class_count)
    # The old 1,000-way classifier is intentionally replaced; all backbone
    # weights are retained for transfer learning.
    compatible = {key: value for key, value in state_dict.items() if not key.startswith('head.')}
    model.load_state_dict(compatible, strict=False)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith('head.')
    return model


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        raise SystemExit('--epochs, --batch-size and --learning-rate must be positive.')
    if not args.input.is_file() or not args.base_checkpoint.is_file():
        raise SystemExit('Input archive or base checkpoint does not exist.')
    with ZipFile(args.input) as archive:
        records, vocabulary = records_from_archive(archive)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.base_checkpoint, len(vocabulary)).to(device)
    loader = DataLoader(DictionaryDataset(args.input, records), batch_size=args.batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for video, target in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(video.to(device))['logits']
            loss = criterion(logits, target.to(device))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(target)
        print(json.dumps({'epoch': epoch + 1, 'loss': total_loss / len(records)}, ensure_ascii=False))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), args.output_dir / 'best.pth')
    (args.output_dir / 'labels.json').write_text(json.dumps({'id_to_label': vocabulary}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'classes': len(vocabulary), 'checkpoint': str(args.output_dir / 'best.pth')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
