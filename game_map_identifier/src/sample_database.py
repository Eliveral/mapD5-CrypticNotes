from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional

import cv2
import numpy as np

from .config import MapConfig
from .deep_matcher import deep_embedding_score
from .door_shape_matcher import door_shape_score
from .matcher import (
    MatchResult,
    MatchSummary,
    anchored_subset_score,
    color_pixel_score,
    corridor_component_score,
    door_anchor_score,
    edge_distance_score,
    entrance_corridor_pair_score,
    entrance_direction_score,
    entrance_patch_score,
    entrance_shape_score,
    partial_structure_score,
    ssim_score,
    template_match_score,
)
from .preprocess import PreprocessConfig, preprocess_image
from .utils import ensure_dir, imread_unicode, imwrite_unicode, timestamp


@dataclass
class ScreenshotSample:
    sample_id: str
    map_id: str
    map_name: str
    image_path: Path
    image: np.ndarray
    preprocessed: np.ndarray
    created_at: str
    note: str = ""
    roi_version: str = ""


class ScreenshotSampleDatabase:
    """User-curated screenshot templates linked to human-confirmed map ids."""

    def __init__(
        self,
        root_dir: Path,
        maps: Mapping[str, MapConfig],
        preprocess_config: PreprocessConfig,
        manifest_path: Optional[Path] = None,
    ) -> None:
        self.root_dir = root_dir
        self.maps = maps
        self.preprocess_config = preprocess_config
        self.manifest_path = manifest_path or root_dir / "config" / "screenshot_samples.json"
        self.sample_dir = root_dir / "assets" / "samples" / "game_roi"
        self.pending_manifest_path = root_dir / "config" / "pending_samples.json"
        self.pending_dir = root_dir / "assets" / "samples" / "pending_roi"
        self.samples: list[ScreenshotSample] = []

    @staticmethod
    def current_roi_version(app_config: Mapping[str, Any]) -> str:
        explicit = str(app_config.get("current_map_feature_roi_version", "")).strip()
        if explicit:
            return explicit
        roi = app_config.get("current_map_feature_roi", [])
        if isinstance(roi, list) and len(roi) == 4:
            return "roi_" + "_".join(str(int(v)) for v in roi)
        return "roi_unknown"

    def load(self) -> "ScreenshotSampleDatabase":
        self.samples = []
        if not self.manifest_path.exists():
            return self
        with self.manifest_path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
        raw_samples = data.get("samples", []) if isinstance(data, dict) else []
        if not isinstance(raw_samples, list):
            return self
        for item in raw_samples:
            if not isinstance(item, dict):
                continue
            rel_image = str(item.get("image", ""))
            map_id = str(item.get("map_id", ""))
            if not rel_image or map_id not in self.maps:
                continue
            path = Path(rel_image)
            if not path.is_absolute():
                path = self.root_dir / path
            image = imread_unicode(path)
            if image is None:
                continue
            pre = preprocess_image(image, self.preprocess_config)
            map_name = str(item.get("map_name") or self.maps[map_id].name)
            self.samples.append(
                ScreenshotSample(
                    sample_id=str(item.get("sample_id") or path.stem),
                    map_id=map_id,
                    map_name=map_name,
                    image_path=path,
                    image=image,
                    preprocessed=pre,
                    created_at=str(item.get("created_at", "")),
                    note=str(item.get("note", "")),
                    roi_version=str(item.get("roi_version", "legacy")),
                )
            )
        return self

    def _read_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {"version": 1, "samples": []}
        with self.manifest_path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"version": 1, "samples": []}
        if not isinstance(data.get("samples"), list):
            data["samples"] = []
        return data

    def save_sample(self, map_id: str, image: np.ndarray, note: str = "", app_config: Optional[Mapping[str, Any]] = None) -> ScreenshotSample:
        if map_id not in self.maps:
            raise ValueError(f"unknown map_id: {map_id}")
        ensure_dir(self.sample_dir)
        ensure_dir(self.manifest_path.parent)
        sample_id = f"{map_id}_{timestamp()}"
        image_path = self.sample_dir / f"{sample_id}.png"
        if not imwrite_unicode(image_path, image):
            raise RuntimeError(f"failed to save sample image: {image_path}")

        rel_path = image_path.relative_to(self.root_dir).as_posix()
        map_name = self.maps[map_id].name
        roi_version = self.current_roi_version(app_config or {})
        item = {
            "sample_id": sample_id,
            "map_id": map_id,
            "map_name": map_name,
            "image": rel_path,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "note": note,
            "roi_version": roi_version,
        }
        manifest = self._read_manifest()
        manifest["version"] = 1
        manifest["samples"].append(item)
        with self.manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
            f.write("\n")

        sample = ScreenshotSample(
            sample_id=sample_id,
            map_id=map_id,
            map_name=map_name,
            image_path=image_path,
            image=image.copy(),
            preprocessed=preprocess_image(image, self.preprocess_config),
            created_at=item["created_at"],
            note=note,
            roi_version=roi_version,
        )
        self.samples.append(sample)
        return sample

    def save_pending_sample(self, image: np.ndarray, note: str = "", app_config: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        ensure_dir(self.pending_dir)
        ensure_dir(self.pending_manifest_path.parent)
        sample_id = f"pending_{timestamp()}"
        image_path = self.pending_dir / f"{sample_id}.png"
        if not imwrite_unicode(image_path, image):
            raise RuntimeError(f"failed to save pending sample image: {image_path}")

        if self.pending_manifest_path.exists():
            with self.pending_manifest_path.open("r", encoding="utf-8-sig") as f:
                manifest = json.load(f)
            if not isinstance(manifest, dict):
                manifest = {"version": 1, "pending_samples": []}
        else:
            manifest = {"version": 1, "pending_samples": []}
        if not isinstance(manifest.get("pending_samples"), list):
            manifest["pending_samples"] = []

        item = {
            "sample_id": sample_id,
            "status": "pending",
            "image": image_path.relative_to(self.root_dir).as_posix(),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "note": note,
            "roi_version": self.current_roi_version(app_config or {}),
        }
        manifest["version"] = 1
        manifest["pending_samples"].append(item)
        with self.pending_manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
            f.write("\n")
        return item

    def pending_count(self) -> int:
        if not self.pending_manifest_path.exists():
            return 0
        with self.pending_manifest_path.open("r", encoding="utf-8-sig") as f:
            manifest = json.load(f)
        if not isinstance(manifest, dict) or not isinstance(manifest.get("pending_samples"), list):
            return 0
        return len(manifest["pending_samples"])

    def match(
        self,
        current_roi: np.ndarray,
        app_config: Mapping[str, Any],
        cnn_scores: Optional[Mapping[str, float]] = None,
    ) -> MatchSummary:
        current_pre = preprocess_image(current_roi, self.preprocess_config)
        cnn_scores = cnn_scores or {}
        sorted_cnn = sorted(cnn_scores.items(), key=lambda item: item[1], reverse=True)
        cnn_rank = {map_id: idx + 1 for idx, (map_id, _score) in enumerate(sorted_cnn)}
        cnn_top_score = float(sorted_cnn[0][1]) if sorted_cnn else 0.0
        cnn_second_score = float(sorted_cnn[1][1]) if len(sorted_cnn) > 1 else 0.0
        cnn_margin = cnn_top_score - cnn_second_score
        cnn_confident = (
            cnn_top_score >= float(app_config.get("sample_cnn_confident_threshold", 0.60))
            and cnn_margin >= float(app_config.get("sample_cnn_confident_margin", 0.25))
        )
        sample_results: list[MatchResult] = []
        use_deep = bool(app_config.get("sample_enable_deep_match", app_config.get("enable_deep_match", True)))
        color_w = float(app_config.get("sample_color_weight", 0.42))
        ssim_w = float(app_config.get("sample_ssim_weight", 0.34))
        template_w = float(app_config.get("sample_template_weight", 0.10))
        deep_w = float(app_config.get("sample_deep_weight", 0.14)) if use_deep else 0.0
        total = max(color_w + ssim_w + template_w + deep_w, 1e-6)
        current_version = self.current_roi_version(app_config)
        mismatch_penalty = float(app_config.get("sample_roi_version_mismatch_penalty", 0.82))

        for sample in self.samples:
            color, color_scale = color_pixel_score(current_roi, sample.image, app_config)
            ss = ssim_score(current_roi, sample.image, app_config)
            tm, tm_scale = template_match_score(current_pre, sample.preprocessed, app_config)
            deep = 0.0
            backend = ""
            if use_deep:
                deep, backend = deep_embedding_score(current_roi, sample.image, app_config)
            final = (color_w * color + ssim_w * ss + template_w * tm + deep_w * deep) / total
            if sample.roi_version and sample.roi_version != current_version:
                final *= mismatch_penalty
            sample_results.append(
                MatchResult(
                    map_id=sample.map_id,
                    map_name=sample.map_name,
                    color_score=color,
                    ssim_score=ss,
                    deep_score=deep,
                    candidate_score=final,
                    template_score=tm,
                    feature_score=0.0,
                    structure_score=0.0,
                    partial_structure_score=0.0,
                    entrance_patch_score=0.0,
                    entrance_shape_score=0.0,
                    entrance_direction_score=0.0,
                    entrance_corridor_pair_score=0.0,
                    anchored_subset_score=0.0,
                    corridor_component_score=0.0,
                    entry_corridor_score=0.0,
                    door_anchor_score=0.0,
                    final_score=final,
                    best_scale=color_scale if color >= tm else tm_scale,
                    deep_backend=backend,
                    sample_id=sample.sample_id,
                    source="sample",
                    roi_version=sample.roi_version,
                    sample_count=sum(1 for item in self.samples if item.map_id == sample.map_id),
                )
            )

        sample_results.sort(key=lambda r: r.candidate_score, reverse=True)
        rerank_n = int(app_config.get("sample_structure_rerank_top_n", 36))
        candidate_map_n = int(app_config.get("sample_structure_candidate_map_top_n", 8))
        rerank_ids = {r.sample_id for r in sample_results[: max(1, rerank_n)]}
        per_map_shape_k = int(app_config.get("sample_door_shape_per_map_top_k", 3))
        per_map_counts: dict[str, int] = {}
        for result in sample_results:
            count = per_map_counts.get(result.map_id, 0)
            if count >= max(1, per_map_shape_k):
                continue
            rerank_ids.add(result.sample_id)
            per_map_counts[result.map_id] = count + 1

        seen_maps: set[str] = set()
        for result in sample_results:
            if result.map_id in seen_maps:
                continue
            seen_maps.add(result.map_id)
            if len(seen_maps) > max(1, candidate_map_n):
                break
            rerank_ids.add(result.sample_id)
        sample_by_id = {sample.sample_id: sample for sample in self.samples}
        visual_w = float(app_config.get("sample_visual_blend_weight", 0.62))
        door_w = float(app_config.get("sample_door_anchor_weight", 0.12))
        structure_w = float(app_config.get("sample_structure_blend_weight", 0.10))
        entrance_w = float(app_config.get("sample_entrance_blend_weight", 0.08))
        corridor_w = float(app_config.get("sample_corridor_blend_weight", 0.08))
        door_shape_w = float(app_config.get("sample_door_shape_weight", 0.34))
        blend_total = max(visual_w + door_w + structure_w + entrance_w + corridor_w, 1e-6)
        door_shape_blend_total = max(visual_w + door_shape_w, 1e-6)

        for result in sample_results:
            if result.sample_id not in rerank_ids:
                continue
            sample = sample_by_id.get(result.sample_id)
            if sample is None:
                continue
            sample_pre = sample.preprocessed
            shape = door_shape_score(current_roi, sample.image, self.preprocess_config)
            result.structure_score = shape.score
            result.partial_structure_score = shape.profile_score
            result.entrance_patch_score = shape.edge_score
            result.entrance_shape_score = shape.patch_score
            result.entrance_direction_score = shape.direction_score
            result.corridor_component_score = shape.branch_score
            if bool(app_config.get("sample_enable_heavy_structure_match", False)):
                result.entrance_patch_score = entrance_patch_score(current_roi, sample.image, app_config, self.preprocess_config)
                result.entrance_shape_score = entrance_shape_score(current_roi, sample.image, app_config, self.preprocess_config)
                result.entrance_direction_score = entrance_direction_score(current_roi, sample.image, app_config, self.preprocess_config)
                result.entrance_corridor_pair_score = entrance_corridor_pair_score(current_roi, sample.image, app_config, self.preprocess_config)
                result.anchored_subset_score = anchored_subset_score(current_roi, sample.image, app_config, self.preprocess_config)
                result.structure_score = max(result.structure_score, edge_distance_score(current_pre, sample_pre))
                result.partial_structure_score = max(result.partial_structure_score, partial_structure_score(current_pre, sample_pre, app_config))
                result.corridor_component_score = max(result.corridor_component_score, corridor_component_score(current_pre, sample_pre))
            icon_anchor_score, _, _ = door_anchor_score(current_roi, sample.image, self.preprocess_config)
            result.door_anchor_score = max(icon_anchor_score, shape.score)
            result.structure_evaluated = True
            structure_blend = 0.46 * result.structure_score + 0.34 * result.partial_structure_score + 0.20 * result.anchored_subset_score
            entrance_blend = (
                0.34 * result.entrance_patch_score
                + 0.26 * result.entrance_shape_score
                + 0.20 * result.entrance_direction_score
                + 0.20 * result.entrance_corridor_pair_score
            )
            legacy_final = (
                visual_w * result.candidate_score
                + door_w * result.door_anchor_score
                + structure_w * structure_blend
                + entrance_w * entrance_blend
                + corridor_w * result.corridor_component_score
            ) / blend_total
            door_shape_final = (visual_w * result.candidate_score + door_shape_w * shape.score) / door_shape_blend_total
            result.final_score = max(legacy_final, door_shape_final)

        for result in sample_results:
            if result.sample_id not in rerank_ids:
                result.final_score = result.candidate_score * float(app_config.get("sample_non_reranked_penalty", 0.92))

        sample_results.sort(key=lambda r: r.final_score, reverse=True)
        by_map: dict[str, list[MatchResult]] = {}
        for result in sample_results:
            by_map.setdefault(result.map_id, []).append(result)

        map_results: list[MatchResult] = []
        support_top_k = int(app_config.get("sample_map_support_top_k", 3))
        support_bonus_weight = float(app_config.get("sample_map_support_bonus_weight", 0.025))
        for map_id, items in by_map.items():
            items.sort(key=lambda r: r.final_score, reverse=True)
            evaluated_items = [r for r in items if r.structure_evaluated]
            best_item = evaluated_items[0] if evaluated_items else items[0]
            best_shape_item = max(evaluated_items, key=lambda r: r.structure_score, default=best_item)
            support = items[: max(1, support_top_k)]
            support_avg = float(np.mean([r.final_score for r in support]))
            door_avg = float(np.mean([r.door_anchor_score for r in support]))
            struct_avg = float(np.mean([max(r.structure_score, r.partial_structure_score, r.anchored_subset_score) for r in support]))
            shape_support = [r for r in items if r.structure_evaluated]
            shape_avg = float(np.mean([r.structure_score for r in shape_support[: max(1, support_top_k)]])) if shape_support else 0.0
            displayed = best_shape_item if best_shape_item.structure_score >= best_item.structure_score else best_item
            displayed_structure = max(best_shape_item.structure_score, struct_avg)
            displayed_door = max(best_shape_item.door_anchor_score, door_avg)
            map_score = (
                0.54 * best_item.final_score
                + 0.18 * support_avg
                + 0.14 * best_shape_item.structure_score
                + 0.06 * shape_avg
                + 0.05 * door_avg
                + 0.03 * struct_avg
            )
            map_score += min(support_bonus_weight, support_bonus_weight * max(0, len(support) - 1) / max(support_top_k - 1, 1))
            cnn_score = float(cnn_scores.get(map_id, 0.0))
            cnn_weight = float(app_config.get("sample_cnn_confident_weight", 0.78)) if cnn_confident else float(app_config.get("sample_cnn_blend_weight", 0.22))
            if cnn_score > 0.0 and cnn_weight > 0.0:
                map_score = (1.0 - cnn_weight) * map_score + cnn_weight * cnn_score
            map_score = min(1.0, map_score)
            map_results.append(
                MatchResult(
                    map_id=map_id,
                    map_name=displayed.map_name,
                    color_score=displayed.color_score,
                    ssim_score=displayed.ssim_score,
                    deep_score=displayed.deep_score,
                    candidate_score=displayed.candidate_score,
                    template_score=displayed.template_score,
                    feature_score=displayed.feature_score,
                    structure_score=displayed_structure,
                    partial_structure_score=displayed.partial_structure_score,
                    entrance_patch_score=displayed.entrance_patch_score,
                    entrance_shape_score=displayed.entrance_shape_score,
                    entrance_direction_score=displayed.entrance_direction_score,
                    entrance_corridor_pair_score=displayed.entrance_corridor_pair_score,
                    anchored_subset_score=displayed.anchored_subset_score,
                    corridor_component_score=displayed.corridor_component_score,
                    entry_corridor_score=0.0,
                    door_anchor_score=displayed_door,
                    final_score=map_score,
                    cnn_score=cnn_score,
                    cnn_rank=cnn_rank.get(map_id, 0),
                    best_scale=displayed.best_scale,
                    is_reliable=False,
                    deep_backend=displayed.deep_backend,
                    sample_id=displayed.sample_id,
                    source="sample_map",
                    roi_version=displayed.roi_version,
                    support_count=len(support),
                    sample_count=len(items),
                )
            )

        map_results.sort(key=lambda r: r.final_score, reverse=True)
        display_n = int(app_config.get("display_top_n", 5))
        top = map_results[:display_n]
        best = top[0] if top else None
        threshold = float(app_config.get("sample_match_threshold", 0.68))
        gap = float(app_config.get("sample_uncertain_gap", 0.035))
        reliable = False
        status = "no screenshot sample matched"
        if best is not None:
            second = top[1].final_score if len(top) > 1 else 0.0
            delta = best.final_score - second
            if cnn_confident and best.cnn_rank == 1:
                reliable = True
                status = (
                    f"cnn confident matched: {best.map_id} / {best.map_name}, "
                    f"cnn={best.cnn_score:.3f}, margin={cnn_margin:.3f}, support={best.support_count}/{best.sample_count}"
                )
            elif best.final_score < threshold:
                status = f"sample match not reliable: top {best.final_score:.3f} < {threshold:.3f}"
            elif delta < gap:
                status = f"sample map match uncertain: gap {delta:.3f} < {gap:.3f}"
            else:
                reliable = True
                status = f"sample map matched: {best.map_id} / {best.map_name} via {best.sample_id}, support={best.support_count}/{best.sample_count}"
            best.is_reliable = reliable
            best.reason = status

        return MatchSummary(
            results=map_results,
            top3=top,
            top_candidates=sample_results[:display_n],
            best=best,
            is_reliable=reliable,
            status=status,
            current_preprocessed=current_pre,
            best_template_preprocessed=best and self._preprocessed_by_sample_id(best.sample_id),
        )

    def _preprocessed_by_sample_id(self, sample_id: str) -> Optional[np.ndarray]:
        for sample in self.samples:
            if sample.sample_id == sample_id:
                return sample.preprocessed
        return None

    def image_by_sample_id(self, sample_id: str) -> Optional[np.ndarray]:
        for sample in self.samples:
            if sample.sample_id == sample_id:
                return sample.image
        return None
