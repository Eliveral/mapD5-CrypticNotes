from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DIFFICULTY_LABELS = tuple(f"map_{index:02d}" for index in range(1, 23))
NIGHTMARE_LABELS = tuple(f"nightmare_{index:02d}" for index in range(1, 14))


@dataclass(frozen=True)
class RecognitionProfile:
    key: str
    display_name: str
    labels: tuple[str, ...]
    manifest_relative_path: str
    sample_dir_relative_path: str
    checkpoint_relative_path: str
    onnx_relative_path: str
    info_relative_path: str
    entry_floor: str
    entry_door: str
    floors: tuple[str, ...]
    architecture: str
    pretrained: bool

    def manifest_path(self, root: Path) -> Path:
        return root / self.manifest_relative_path

    def sample_dir(self, root: Path) -> Path:
        return root / self.sample_dir_relative_path

    def checkpoint_path(self, root: Path) -> Path:
        return root / self.checkpoint_relative_path

    def onnx_path(self, root: Path) -> Path:
        return root / self.onnx_relative_path

    def info_path(self, root: Path) -> Path:
        return root / self.info_relative_path

    def label_name(self, label: str) -> str:
        if label not in self.labels:
            raise ValueError(f"label {label!r} does not belong to profile {self.key!r}")
        number = int(label.rsplit("_", 1)[1])
        return f"{self.display_name}布局 {number:02d}"


DIFFICULTY_PROFILE = RecognitionProfile(
    key="difficulty",
    display_name="困难模式",
    labels=DIFFICULTY_LABELS,
    manifest_relative_path="config/screenshot_samples.json",
    sample_dir_relative_path="assets/samples/game_roi",
    checkpoint_relative_path="models/map_cnn.pt",
    onnx_relative_path="models/map_cnn.onnx",
    info_relative_path="models/map_cnn_info.json",
    entry_floor="1st",
    entry_door="first_floor_side_door",
    floors=("1st", "2nd"),
    architecture="resnet18",
    pretrained=True,
)

NIGHTMARE_PROFILE = RecognitionProfile(
    key="nightmare",
    display_name="噩梦模式",
    labels=NIGHTMARE_LABELS,
    manifest_relative_path="config/nightmare_screenshot_samples.json",
    sample_dir_relative_path="assets/samples/nightmare_game_roi",
    checkpoint_relative_path="models/nightmare_map_cnn.pt",
    onnx_relative_path="models/nightmare_map_cnn.onnx",
    info_relative_path="models/nightmare_map_cnn_info.json",
    entry_floor="2nd",
    entry_door="second_floor_side_door",
    floors=("b1", "1st", "2nd"),
    architecture="resnet18",
    pretrained=True,
)

RECOGNITION_PROFILES = {
    DIFFICULTY_PROFILE.key: DIFFICULTY_PROFILE,
    NIGHTMARE_PROFILE.key: NIGHTMARE_PROFILE,
}

_PROFILE_ALIASES = {
    "difficult": "difficulty",
    "hard": "difficulty",
    "nightmare_mode": "nightmare",
}


def get_recognition_profile(value: str | None) -> RecognitionProfile:
    key = str(value or "difficulty").strip().lower()
    key = _PROFILE_ALIASES.get(key, key)
    try:
        return RECOGNITION_PROFILES[key]
    except KeyError as exc:
        choices = ", ".join(sorted(RECOGNITION_PROFILES))
        raise ValueError(f"unknown recognition profile {value!r}; choose one of: {choices}") from exc
