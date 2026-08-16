"""Export the exact classifier head used by camera.ipynb to ONNX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_camera_model(notebook_path: Path, checkpoint_path: Path) -> torch.nn.Module:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    # Cell 1 also imports OpenCV and MediaPipe, neither of which is needed to
    # reconstruct the network and may be unavailable in an export environment.
    # Provide precisely the symbols used by cell 2, then execute its exact
    # architecture definition from camera.ipynb.
    namespace: dict[str, object] = {
        "__name__": "vsl30_camera_export",
        "np": np,
        "torch": torch,
        "nn": nn,
        "F": F,
    }
    exec("".join(notebook["cells"][2]["source"]), namespace)  # noqa: S102
    model_type = namespace["VSL30Model"]
    encoder_type = namespace["KeypointEncoder30"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = model_type(encoder_type(), num_classes=len(checkpoint["target_glosses"]))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.eval()


class LogitsOnly(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, keypoints: torch.Tensor) -> torch.Tensor:
        return self.model(keypoints)["logits"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = LogitsOnly(load_camera_model(args.notebook, args.checkpoint)).eval()
    dummy = torch.zeros((1, 48, 75, 4), dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["keypoints"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print(args.output)


if __name__ == "__main__":
    main()
