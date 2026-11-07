from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

from .utils import imread_unicode


DEFAULT_IMAGE_SIZE = (48, 32)


@dataclass(frozen=True)
class MLPModelInfo:
    labels: list[str]
    image_size: tuple[int, int]
    mean: list[float]
    std: list[float]


def _as_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def extract_mlp_features(image: np.ndarray, image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    """Extract a compact full-ROI feature vector for the first neural classifier.

    This intentionally keeps the first version simple and dependency-free:
    grayscale texture + edge structure from the whole feature ROI.
    """
    bgr = _as_bgr(image)
    resized = cv2.resize(bgr, image_size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.equalizeHist(gray)
    edges = cv2.Canny(gray, 45, 135)
    gray_f = gray.astype(np.float32) / 255.0
    edge_f = edges.astype(np.float32) / 255.0
    feature = np.concatenate([gray_f.reshape(-1), edge_f.reshape(-1)]).astype(np.float32)
    return feature


def augment_image(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Small realistic augmentations for captured minimap ROIs."""
    bgr = _as_bgr(image).copy()
    h, w = bgr.shape[:2]
    scale = float(rng.uniform(0.96, 1.04))
    tx = int(rng.integers(-max(1, w // 80), max(2, w // 80 + 1)))
    ty = int(rng.integers(-max(1, h // 80), max(2, h // 80 + 1)))
    mat = np.array([[scale, 0.0, tx + (1.0 - scale) * w / 2.0], [0.0, scale, ty + (1.0 - scale) * h / 2.0]], dtype=np.float32)
    out = cv2.warpAffine(bgr, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)

    alpha = float(rng.uniform(0.92, 1.08))
    beta = float(rng.uniform(-8.0, 8.0))
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)
    if rng.random() < 0.30:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    return out


def standardize_features(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / np.maximum(std, 1e-6)).astype(np.float32)


def softmax(scores: np.ndarray) -> np.ndarray:
    scores = scores.astype(np.float64)
    scores -= np.max(scores)
    exp = np.exp(scores)
    return (exp / max(float(exp.sum()), 1e-12)).astype(np.float32)


def load_model_info(path: Path) -> MLPModelInfo:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return MLPModelInfo(
        labels=[str(v) for v in data["labels"]],
        image_size=(int(data["image_size"][0]), int(data["image_size"][1])),
        mean=[float(v) for v in data["mean"]],
        std=[float(v) for v in data["std"]],
    )


def save_model_info(path: Path, info: MLPModelInfo, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "labels": info.labels,
        "image_size": list(info.image_size),
        "mean": info.mean,
        "std": info.std,
    }
    if extra:
        payload.update(extra)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


class OpenCVMLPClassifier:
    def __init__(self, model_path: Path, info_path: Path) -> None:
        self.model_path = model_path
        self.info = load_model_info(info_path)
        self.model = cv2.ml.ANN_MLP_load(str(model_path))
        if self.model.empty():
            raise RuntimeError(f"failed to load MLP model: {model_path}")
        self.mean = np.asarray(self.info.mean, dtype=np.float32)
        self.std = np.asarray(self.info.std, dtype=np.float32)

    def predict_proba(self, image: np.ndarray) -> list[tuple[str, float]]:
        feature = extract_mlp_features(image, self.info.image_size)[None, :]
        feature = standardize_features(feature, self.mean, self.std)
        _ret, response = self.model.predict(feature)
        probs = softmax(response.reshape(-1))
        pairs = [(label, float(probs[i])) for i, label in enumerate(self.info.labels)]
        return sorted(pairs, key=lambda item: item[1], reverse=True)


def read_labelled_samples(root: Path, manifest_path: Path) -> list[tuple[str, Path]]:
    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    out: list[tuple[str, Path]] = []
    for item in data.get("samples", []):
        if not isinstance(item, dict):
            continue
        map_id = str(item.get("map_id", ""))
        image_raw = str(item.get("image", ""))
        if not map_id or not image_raw:
            continue
        image_path = Path(image_raw)
        if not image_path.is_absolute():
            image_path = root / image_path
        if image_path.exists():
            out.append((map_id, image_path))
    return out
