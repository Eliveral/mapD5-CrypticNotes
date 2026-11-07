from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from typing import Any, Mapping

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


def _normalize_vector(values: np.ndarray) -> np.ndarray:
    vec = values.astype(np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        return vec
    return vec / norm


def _cosine01(a: np.ndarray, b: np.ndarray) -> float:
    aa = _normalize_vector(a)
    bb = _normalize_vector(b)
    if aa.size == 0 or bb.size == 0 or aa.size != bb.size:
        return 0.0
    cosine = float(np.dot(aa, bb))
    return max(0.0, min(1.0, (cosine + 1.0) * 0.5))


def handcrafted_embedding(image: np.ndarray, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Small local descriptor used when a neural backend is unavailable.

    It is not a neural network. It exists as a practical fallback so the new
    reranking pipeline keeps working while a torch/DINOv2 backend is optional.
    """
    if image.size == 0:
        return np.zeros((1,), dtype=np.float32)
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = image
    bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    features: list[np.ndarray] = []
    for grid in (4, 7):
        cell_h = size[1] // grid
        cell_w = size[0] // grid
        for y in range(grid):
            for x in range(grid):
                y0 = y * cell_h
                x0 = x * cell_w
                y1 = size[1] if y == grid - 1 else (y + 1) * cell_h
                x1 = size[0] if x == grid - 1 else (x + 1) * cell_w
                lab_patch = lab[y0:y1, x0:x1]
                hsv_patch = hsv[y0:y1, x0:x1]
                edge_patch = edges[y0:y1, x0:x1]
                features.append(lab_patch.mean(axis=(0, 1)) / 255.0)
                features.append(lab_patch.std(axis=(0, 1)) / 255.0)
                hist = cv2.calcHist([hsv_patch], [0, 1], None, [8, 6], [0, 180, 0, 256])
                cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
                features.append(hist.reshape(-1))
                features.append(np.array([float((edge_patch > 0).mean())], dtype=np.float32))

    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    dct = cv2.dct(small)[:10, :10].reshape(-1)
    features.append(dct)
    return _normalize_vector(np.concatenate(features).astype(np.float32))


def handcrafted_embedding_score(current: np.ndarray, template: np.ndarray) -> float:
    return _cosine01(handcrafted_embedding(current), handcrafted_embedding(template))


def _image_hash(image: np.ndarray) -> str:
    if image.size == 0:
        return "empty"
    digest = hashlib.sha1()
    digest.update(str(image.shape).encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _load_dinov2_model(model_name: str):
    import torch

    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval()
    return model


@lru_cache(maxsize=128)
def _dinov2_embedding_from_hash(image_hash: str, shape: tuple[int, int, int], data: bytes, model_name: str) -> np.ndarray:
    import torch

    image = np.frombuffer(data, dtype=np.uint8).reshape(shape)
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = (rgb - mean) / std
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
    model = _load_dinov2_model(model_name)
    with torch.no_grad():
        output = model(tensor)
    if isinstance(output, dict):
        output = output.get("x_norm_clstoken") or output.get("last_hidden_state") or next(iter(output.values()))
    arr = output.detach().cpu().numpy().reshape(-1).astype(np.float32)
    return _normalize_vector(arr)


def dinov2_embedding_score(current: np.ndarray, template: np.ndarray, app_config: Mapping[str, Any]) -> tuple[float, str]:
    model_name = str(app_config.get("deep_match_model", "dinov2_vits14"))
    cur = np.ascontiguousarray(current)
    tpl = np.ascontiguousarray(template)
    cur_emb = _dinov2_embedding_from_hash(_image_hash(cur), tuple(cur.shape), cur.tobytes(), model_name)
    tpl_emb = _dinov2_embedding_from_hash(_image_hash(tpl), tuple(tpl.shape), tpl.tobytes(), model_name)
    return _cosine01(cur_emb, tpl_emb), f"dinov2:{model_name}"


_DEEP_BACKEND_FAILED = False
_DEEP_BACKEND_REASON = ""


def deep_embedding_score(current: np.ndarray, template: np.ndarray, app_config: Mapping[str, Any]) -> tuple[float, str]:
    """Return a neural/embedding rerank score for a candidate pair.

    The preferred backend is DINOv2 via torch.hub. If torch/model loading is not
    available and fallback is enabled, it returns a handcrafted embedding score
    instead of failing the recognition loop.
    """
    global _DEEP_BACKEND_FAILED, _DEEP_BACKEND_REASON
    if current.size == 0 or template.size == 0:
        return 0.0, "empty"

    backend = str(app_config.get("deep_match_backend", "dinov2_torchhub")).lower()
    if backend in {"dinov2", "dinov2_torchhub", "auto"} and not _DEEP_BACKEND_FAILED:
        try:
            return dinov2_embedding_score(current, template, app_config)
        except Exception as exc:  # pragma: no cover - optional heavy dependency.
            _DEEP_BACKEND_FAILED = True
            _DEEP_BACKEND_REASON = f"{type(exc).__name__}: {exc}"
            LOGGER.warning("DINOv2 backend unavailable, falling back if allowed: %s", _DEEP_BACKEND_REASON)

    if bool(app_config.get("deep_fallback_to_handcrafted_embedding", True)):
        return handcrafted_embedding_score(current, template), "handcrafted"
    return 0.0, f"unavailable:{_DEEP_BACKEND_REASON or backend}"
