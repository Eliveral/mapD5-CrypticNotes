from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional

import cv2
import numpy as np

from .config import MapConfig, ProjectConfig
from .preprocess import PreprocessConfig, preprocess_image
from .utils import crop_roi, imread_unicode


class MapDatabaseError(RuntimeError):
    """Raised when map assets cannot be loaded."""


@dataclass
class MapTemplate:
    map_id: str
    name: str
    full_image: np.ndarray
    template: np.ndarray
    preprocessed: np.ndarray
    keypoints: tuple
    descriptors: Optional[np.ndarray]
    feature_roi: tuple[int, int, int, int]
    template_roi_key: str


class MapDatabase:
    def __init__(
        self,
        maps: Mapping[str, MapConfig],
        preprocess_config: PreprocessConfig,
        orb: Optional[cv2.ORB] = None,
    ) -> None:
        self.maps = maps
        self.preprocess_config = preprocess_config
        self.orb = orb
        self.templates: Dict[str, MapTemplate] = {}

    def load(self, strict: bool = False) -> "MapDatabase":
        errors: list[str] = []
        for map_id, cfg in self.maps.items():
            image = imread_unicode(cfg.full_image)
            if image is None:
                msg = f"{map_id} 地图文件缺失或无法读取：{cfg.full_image}"
                if strict:
                    errors.append(msg)
                continue
            try:
                template = crop_roi(image, cfg.feature_roi, f"{map_id}.feature_roi")
            except Exception as exc:
                msg = f"{map_id} ROI 无效：{cfg.feature_roi}，{exc}"
                if strict:
                    errors.append(msg)
                continue

            pre = preprocess_image(template, self.preprocess_config)
            keypoints = ()
            descriptors = None
            if self.orb is not None:
                kp, des = self.orb.detectAndCompute(pre, None)
                keypoints = tuple(kp or ())
                descriptors = des
            self.templates[map_id] = MapTemplate(
                map_id=map_id,
                name=cfg.name,
                full_image=image,
                template=template,
                preprocessed=pre,
                keypoints=keypoints,
                descriptors=descriptors,
                feature_roi=cfg.feature_roi,
                template_roi_key=cfg.template_roi_key,
            )
        if strict and errors:
            raise MapDatabaseError("\n".join(errors))
        if not self.templates:
            detail = "\n".join(errors) if errors else "没有可用地图模板"
            raise MapDatabaseError(f"没有加载到任何地图模板。请检查 assets/maps/full 和 maps.json。\n{detail}")
        return self


def create_orb_from_config(app_config: Mapping) -> Optional[cv2.ORB]:
    if not bool(app_config.get("enable_orb", True)):
        return None
    orb_cfg = app_config.get("orb", {}) or {}
    return cv2.ORB_create(
        nfeatures=int(orb_cfg.get("nfeatures", 1200)),
        scaleFactor=float(orb_cfg.get("scaleFactor", 1.2)),
        nlevels=int(orb_cfg.get("nlevels", 8)),
    )
