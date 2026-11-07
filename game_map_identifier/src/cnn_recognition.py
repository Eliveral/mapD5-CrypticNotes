from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .cnn_classifier import OpenCVDNNClassifier, TorchCNNClassifier
from .cnn_sample_capture import capture_scaled_cnn_recognition_roi
from .recognition_profiles import RecognitionProfile, get_recognition_profile


class CNNRecognitionService:
    """Profile-aware CNN inference shared by runtime and offline tooling."""

    def __init__(self, root: Path, profile: RecognitionProfile, *, device: str | None = None) -> None:
        self.root = root
        self.profile = profile
        onnx_path = profile.onnx_path(root)
        checkpoint_path = profile.checkpoint_path(root)
        info_path = profile.info_path(root)
        if onnx_path.exists() and info_path.exists():
            self.classifier = OpenCVDNNClassifier(onnx_path, info_path)
        elif checkpoint_path.exists() and info_path.exists():
            self.classifier = TorchCNNClassifier(checkpoint_path, info_path, device=device)
        else:
            raise RuntimeError(
                f"missing {profile.display_name} CNN model; expected {onnx_path} and {info_path}, "
                f"or {checkpoint_path} and {info_path}"
            )
        model_labels = tuple(self.classifier.info.labels)
        if model_labels != profile.labels:
            raise RuntimeError(
                f"CNN model labels do not match profile {profile.key!r}: "
                f"expected {list(profile.labels)}, got {list(model_labels)}"
            )

    @classmethod
    def from_name(cls, root: Path, profile_name: str | None, *, device: str | None = None) -> "CNNRecognitionService":
        return cls(root, get_recognition_profile(profile_name), device=device)

    def predict_proba(self, image: np.ndarray) -> list[tuple[str, float]]:
        return self.classifier.predict_proba(image)

    def capture_and_predict(self, app_config: Mapping[str, Any] | None = None):
        capture, roi = capture_scaled_cnn_recognition_roi(app_config)
        return capture, roi, self.predict_proba(roi)

