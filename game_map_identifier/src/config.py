from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple


ROI = Tuple[int, int, int, int]


class ConfigError(RuntimeError):
    """Raised when a configuration file is missing or invalid."""


@dataclass(frozen=True)
class MapConfig:
    map_id: str
    name: str
    full_image: Path
    feature_roi: ROI
    template_roi_key: str


@dataclass(frozen=True)
class ProjectConfig:
    root_dir: Path
    app_config_path: Path
    maps_config_path: Path
    app: Dict[str, Any]
    maps: Dict[str, MapConfig]


def project_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"配置文件不存在：{path}")
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"JSON 格式错误：{path}，{exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"配置文件根节点必须是对象：{path}")
    return data


def load_app_config_for_first_run(app_config_path: Optional[Path] = None) -> tuple[Path, Dict[str, Any]]:
    """Read only the app configuration before first-run ROI fields exist."""
    root = project_root()
    path = app_config_path or root / "config" / "config.json"
    return path, _read_json(path)


def _parse_roi(value: Any, field_name: str) -> ROI:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or not all(isinstance(v, (int, float)) for v in value)
    ):
        raise ConfigError(f"{field_name} 必须是 [x, y, w, h] 四个数字")
    x, y, w, h = (int(v) for v in value)
    if w <= 0 or h <= 0:
        raise ConfigError(f"{field_name} 的宽高必须大于 0：{value}")
    return x, y, w, h


def resolve_project_path(root_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root_dir / path


def load_project_config(
    root_dir: Optional[Path] = None,
    app_config_path: Optional[Path] = None,
    maps_config_path: Optional[Path] = None,
) -> ProjectConfig:
    root = root_dir or project_root()
    app_path = app_config_path or root / "config" / "config.json"
    maps_path = maps_config_path or root / "config" / "maps.json"

    app = _read_json(app_path)
    raw_maps = _read_json(maps_path)

    # Validate the ROI fields that are required for runtime.
    _parse_roi(app.get("screen_capture_roi"), "screen_capture_roi")
    _parse_roi(app.get("window_relative_map_roi"), "window_relative_map_roi")
    _parse_roi(app.get("current_map_feature_roi"), "current_map_feature_roi")
    if app.get("alignment_screen_capture_roi") is not None:
        _parse_roi(app.get("alignment_screen_capture_roi"), "alignment_screen_capture_roi")
    if app.get("alignment_map_roi") is not None:
        _parse_roi(app.get("alignment_map_roi"), "alignment_map_roi")
    template_roi_key = str(app.get("template_roi_key", "door_1_corridor_roi"))
    raw_fallback_keys = app.get("template_roi_fallback_keys", ["door_1_roi", "feature_roi"])
    if not isinstance(raw_fallback_keys, list):
        raw_fallback_keys = ["door_1_roi", "feature_roi"]
    fallback_keys = [str(k) for k in raw_fallback_keys if str(k)]
    candidate_roi_keys = [template_roi_key] + [k for k in fallback_keys if k != template_roi_key]

    capture_mode = app.get("capture_mode", "fixed_screen_roi")
    if capture_mode not in {"fixed_screen_roi", "window_relative"}:
        raise ConfigError("capture_mode 必须是 fixed_screen_roi 或 window_relative")

    maps: Dict[str, MapConfig] = {}
    for map_id, item in raw_maps.items():
        if not isinstance(item, Mapping):
            raise ConfigError(f"地图配置 {map_id} 必须是对象")
        full_image = item.get("full_image")
        if not full_image:
            raise ConfigError(f"地图 {map_id} 缺少 full_image")
        selected_roi_key = next((key for key in candidate_roi_keys if item.get(key) is not None), "")
        if not selected_roi_key:
            raise ConfigError(
                f"地图 {map_id} 缺少模板 ROI。请至少配置 {template_roi_key}，"
                f"或回退字段之一：{', '.join(fallback_keys)}"
            )
        roi = _parse_roi(item.get(selected_roi_key), f"{map_id}.{selected_roi_key}")
        maps[map_id] = MapConfig(
            map_id=map_id,
            name=str(item.get("name", map_id)),
            full_image=resolve_project_path(root, str(full_image)),
            feature_roi=roi,
            template_roi_key=selected_roi_key,
        )

    if not maps:
        raise ConfigError("maps.json 中没有任何地图配置")

    return ProjectConfig(root, app_path, maps_path, app, maps)


def save_maps_json(root_dir: Path, maps: Mapping[str, Mapping[str, Any]], path: Optional[Path] = None) -> None:
    target = path or root_dir / "config" / "maps.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(maps, f, ensure_ascii=False, indent=2)
        f.write("\n")


def save_app_config(path: Path, app_config: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(app_config, f, ensure_ascii=False, indent=2)
        f.write("\n")


def load_raw_maps_json(root_dir: Optional[Path] = None, path: Optional[Path] = None) -> Dict[str, Any]:
    root = root_dir or project_root()
    return _read_json(path or root / "config" / "maps.json")


def roi_from_app(config: Mapping[str, Any], key: str) -> ROI:
    return _parse_roi(config.get(key), key)
