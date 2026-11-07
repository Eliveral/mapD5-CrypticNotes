from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    import torch
    from torch import nn
    from torchvision import models
except Exception:  # pragma: no cover - torch is optional in packaged runtime.
    torch = None
    nn = None
    models = None

from .utils import imread_unicode


DEFAULT_CNN_IMAGE_SIZE = (192, 128)
DEFAULT_CNN_ARCHITECTURE = "small_cnn"


@dataclass(frozen=True)
class CNNModelInfo:
    labels: list[str]
    image_size: tuple[int, int]
    mean: list[float]
    std: list[float]
    architecture: str = DEFAULT_CNN_ARCHITECTURE


if nn is not None:
    class SmallMapCNN(nn.Module):
        def __init__(self, num_classes: int) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 24, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(24),
                nn.SiLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(48),
                nn.SiLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(48, 96, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(96),
                nn.SiLU(inplace=True),
                nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(96),
                nn.SiLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(96, 160, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(160),
                nn.SiLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.25),
                nn.Linear(160, 96),
                nn.SiLU(inplace=True),
                nn.Dropout(0.15),
                nn.Linear(96, num_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))


    def create_cnn_model(num_classes: int, architecture: str = DEFAULT_CNN_ARCHITECTURE, pretrained: bool = False) -> nn.Module:
        arch = str(architecture or DEFAULT_CNN_ARCHITECTURE).lower()
        if arch in {"small", "small_cnn", "smallmapcnn"}:
            return SmallMapCNN(num_classes)
        if arch == "resnet18":
            weights = None
            if pretrained and models is not None:
                try:
                    weights = models.ResNet18_Weights.DEFAULT
                except Exception:
                    weights = None
            if models is None:
                raise RuntimeError("torchvision is required for resnet18")
            model = models.resnet18(weights=weights)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model
        raise ValueError(f"unsupported CNN architecture: {architecture}")
else:
    class SmallMapCNN:
        def __init__(self, _num_classes: int) -> None:
            raise RuntimeError("PyTorch is not available")


    def create_cnn_model(_num_classes: int, architecture: str = DEFAULT_CNN_ARCHITECTURE, pretrained: bool = False):
        raise RuntimeError("PyTorch is not available")


def read_labelled_samples(root: Path, manifest_path: Path) -> list[tuple[str, Path]]:
    with manifest_path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    out: list[tuple[str, Path]] = []
    for item in data.get("samples", []):
        if not isinstance(item, dict):
            continue
        label = str(item.get("map_id", ""))
        image_raw = str(item.get("image", ""))
        if not label or not image_raw:
            continue
        path = Path(image_raw)
        if not path.is_absolute():
            path = root / path
        if path.exists():
            out.append((label, path))
    return out


def preprocess_for_cnn_array(
    image: np.ndarray,
    image_size: tuple[int, int],
    mean: list[float] | np.ndarray,
    std: list[float] | np.ndarray,
) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    resized = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    arr = np.transpose(rgb, (2, 0, 1))
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
    std_arr = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
    arr = (arr - mean_arr) / np.maximum(std_arr, 1e-6)
    return arr.astype(np.float32)


def preprocess_for_cnn(
    image: np.ndarray,
    image_size: tuple[int, int],
    mean: list[float] | np.ndarray,
    std: list[float] | np.ndarray,
):
    if torch is None:
        raise RuntimeError("PyTorch is not available")
    arr = preprocess_for_cnn_array(image, image_size, mean, std)
    return torch.from_numpy(arr.astype(np.float32))


def save_cnn_info(path: Path, info: CNNModelInfo, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "labels": info.labels,
        "image_size": list(info.image_size),
        "mean": info.mean,
        "std": info.std,
        "architecture": info.architecture,
    }
    if extra:
        payload.update(extra)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def load_cnn_info(path: Path) -> CNNModelInfo:
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    return CNNModelInfo(
        labels=[str(v) for v in data["labels"]],
        image_size=(int(data["image_size"][0]), int(data["image_size"][1])),
        mean=[float(v) for v in data["mean"]],
        std=[float(v) for v in data["std"]],
        architecture=str(data.get("architecture", DEFAULT_CNN_ARCHITECTURE)),
    )


class TorchCNNClassifier:
    def __init__(self, checkpoint_path: Path, info_path: Path, device: str | None = None) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is not available")
        self.info = load_cnn_info(info_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = create_cnn_model(len(self.info.labels), self.info.architecture, pretrained=False).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state = checkpoint.get("model_state", checkpoint)
        self.model.load_state_dict(state)
        self.model.eval()

    def predict_proba(self, image: np.ndarray) -> list[tuple[str, float]]:
        if torch is None:
            raise RuntimeError("PyTorch is not available")
        tensor = preprocess_for_cnn(image, self.info.image_size, self.info.mean, self.info.std)
        tensor = tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
        pairs = [(label, float(probs[idx])) for idx, label in enumerate(self.info.labels)]
        return sorted(pairs, key=lambda item: item[1], reverse=True)


class OpenCVDNNClassifier:
    def __init__(self, onnx_path: Path, info_path: Path) -> None:
        self.info = load_cnn_info(info_path)
        self._temp_onnx_path: Path | None = None
        self.net = self._read_onnx_with_ascii_path(onnx_path)

    def _read_onnx_with_ascii_path(self, onnx_path: Path):
        path = onnx_path.resolve()
        try:
            str(path).encode("ascii")
            return cv2.dnn.readNetFromONNX(str(path))
        except UnicodeEncodeError:
            temp_dir = Path(tempfile.gettempdir()) / "GameMapIdentifier"
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_path = temp_dir / f"{path.stem}_{path.stat().st_mtime_ns}.onnx"
            if not temp_path.exists() or temp_path.stat().st_size != path.stat().st_size:
                shutil.copy2(path, temp_path)
            self._temp_onnx_path = temp_path
            return cv2.dnn.readNetFromONNX(str(temp_path))

    def predict_proba(self, image: np.ndarray) -> list[tuple[str, float]]:
        blob = preprocess_for_cnn_array(image, self.info.image_size, self.info.mean, self.info.std)[None, ...]
        self.net.setInput(blob)
        logits = self.net.forward().reshape(-1).astype(np.float32)
        logits = logits - float(np.max(logits))
        exp = np.exp(logits)
        probs = exp / max(float(np.sum(exp)), 1e-8)
        pairs = [(label, float(probs[idx])) for idx, label in enumerate(self.info.labels)]
        return sorted(pairs, key=lambda item: item[1], reverse=True)


def read_image(path: Path) -> np.ndarray:
    image = imread_unicode(path)
    if image is None:
        raise RuntimeError(f"failed to read image: {path}")
    return image
