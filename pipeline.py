#!/usr/bin/env python3
"""Unified slide processing for clahe, stain, and nuclei outputs."""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import resource
import time
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import histomicstk as htk
import histomicstk.segmentation.positive_pixel_count as ppc
import large_image
import numpy as np
import openslide
import pandas as pd
from histomicstk.saliency.tissue_detection import get_tissue_mask
from skimage import measure, morphology
import scipy as sp


WSI_EXTS = {".svs", ".tif", ".tiff", ".ndpi", ".vms", ".vmu", ".scn", ".mrxs", ".czi"}
DEFAULT_REFERENCE_IMAGE = (
    "/well/parkkinen/users/nfw313/nucleidensityapprox/ref_image_reinhard/"
    "ref_tile_00-1027-A-IBA1.npy"
)
_worker_ppc_params: ppc.Parameters | None = None
_worker_mean_ref: np.ndarray | None = None
_worker_std_ref: np.ndarray | None = None
_worker_stain_matrix: np.ndarray | None = None


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} [histoprocessor] {message}", flush=True)


def _worker_init(
    ppc_params_dict: dict[str, Any] | None,
    mean_ref: np.ndarray | None,
    std_ref: np.ndarray | None,
    stain_matrix: np.ndarray | None,
) -> None:
    global _worker_ppc_params, _worker_mean_ref, _worker_std_ref, _worker_stain_matrix
    _worker_ppc_params = ppc.Parameters(**ppc_params_dict) if ppc_params_dict else None
    _worker_mean_ref = mean_ref
    _worker_std_ref = std_ref
    _worker_stain_matrix = stain_matrix


def pad_center_2d(arr: np.ndarray, target_hw=(1024, 1024), pad_value=0) -> np.ndarray:
    h, w = arr.shape[:2]
    th, tw = target_hw
    ph = max(0, th - h)
    pw = max(0, tw - w)
    top = ph // 2
    bottom = ph - top
    left = pw // 2
    right = pw - left
    return np.pad(arr, ((top, bottom), (left, right)), mode="constant", constant_values=pad_value)


def fit_and_pad_2d(
    arr: np.ndarray,
    target_hw: tuple[int, int] = (1024, 1024),
    *,
    interpolation: int = cv2.INTER_NEAREST,
    pad_value: int | float = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    h, w = arr.shape[:2]
    th, tw = target_hw
    if h <= 0 or w <= 0:
        raise ValueError("Cannot fit-and-pad an empty array")

    scale = min(1.0, th / float(h), tw / float(w))
    if scale < 1.0:
        resized_w = max(1, int(round(w * scale)))
        resized_h = max(1, int(round(h * scale)))
        resized = cv2.resize(arr, (resized_w, resized_h), interpolation=interpolation)
    else:
        resized = arr
        resized_h, resized_w = h, w

    ph = max(0, th - resized_h)
    pw = max(0, tw - resized_w)
    top = ph // 2
    bottom = ph - top
    left = pw // 2
    right = pw - left
    padded = np.pad(resized, ((top, bottom), (left, right)), mode="constant", constant_values=pad_value)
    meta = {
        "target_height": int(th),
        "target_width": int(tw),
        "source_height": int(h),
        "source_width": int(w),
        "embedded_height": int(resized_h),
        "embedded_width": int(resized_w),
        "pad_top": int(top),
        "pad_bottom": int(bottom),
        "pad_left": int(left),
        "pad_right": int(right),
        "crop_y0": int(top),
        "crop_y1": int(top + resized_h),
        "crop_x0": int(left),
        "crop_x1": int(left + resized_w),
        "scale": float(scale),
    }
    return padded, meta


def apply_gamma(image_u8: np.ndarray, gamma=1.0) -> np.ndarray:
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)], dtype=np.uint8)
    return cv2.LUT(image_u8, table)


def maybe_strip_alpha(tile_np: np.ndarray) -> np.ndarray:
    if tile_np.ndim == 3 and tile_np.shape[-1] == 4:
        return tile_np[..., :3]
    return tile_np


def build_thumbnail_tissue_mask(
    slide_path: Path,
    *,
    gamma: float = 0.5,
    clahe_clip: float = 12.0,
    clahe_grid: int = 4,
    dilate_kernel: int = 31,
    close_kernel: int = 9,
) -> tuple[np.ndarray, tuple[int, int]]:
    slide = openslide.OpenSlide(str(slide_path))
    thumb = np.array(slide.associated_images["thumbnail"])
    img_gray = cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY)
    raw_labeled, _ = get_tissue_mask(255 - img_gray)
    raw_mask = raw_labeled > 0
    gamma_corrected = apply_gamma(img_gray, gamma=gamma)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
    enhanced = clahe.apply(gamma_corrected)
    enhanced_labeled, _ = get_tissue_mask(255 - enhanced)
    enhanced_mask = enhanced_labeled > 0
    mask_u8 = np.logical_or(raw_mask, enhanced_mask).astype(np.uint8)
    if close_kernel > 1:
        kernel = np.ones((close_kernel, close_kernel), dtype=np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    if dilate_kernel > 1:
        kernel = np.ones((dilate_kernel, dilate_kernel), dtype=np.uint8)
        mask_u8 = cv2.dilate(mask_u8, kernel, iterations=1)
    return mask_u8.astype(bool), (thumb.shape[1], thumb.shape[0])


def build_tile_keep_grid(
    tissue_mask: np.ndarray,
    thumb_size: tuple[int, int],
    slide_size: tuple[int, int],
    tile_size: int,
    overlap_threshold: float = 0.0,
    dilate_tiles: int = 1,
) -> np.ndarray:
    thumb_w, thumb_h = thumb_size
    slide_w, slide_h = slide_size
    tiles_x = int(np.ceil(slide_w / max(1, tile_size)))
    tiles_y = int(np.ceil(slide_h / max(1, tile_size)))
    keep = np.zeros((tiles_y, tiles_x), dtype=bool)

    for ty in range(tiles_y):
        gy = ty * tile_size
        gh = min(tile_size, max(0, slide_h - gy))
        for tx in range(tiles_x):
            gx = tx * tile_size
            gw = min(tile_size, max(0, slide_w - gx))

            x0 = max(0, min(thumb_w - 1, int(np.floor(gx * thumb_w / max(1, slide_w)))))
            y0 = max(0, min(thumb_h - 1, int(np.floor(gy * thumb_h / max(1, slide_h)))))
            x1 = max(x0 + 1, min(thumb_w, int(np.ceil((gx + gw) * thumb_w / max(1, slide_w)))))
            y1 = max(y0 + 1, min(thumb_h, int(np.ceil((gy + gh) * thumb_h / max(1, slide_h)))))

            region = tissue_mask[y0:y1, x0:x1]
            overlap = float(region.mean()) if region.size else 0.0
            keep[ty, tx] = overlap > overlap_threshold

    if dilate_tiles > 0:
        kernel_size = 2 * int(dilate_tiles) + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        keep = cv2.dilate(keep.astype(np.uint8), kernel, iterations=1).astype(bool)

    return keep


def should_keep_tile(tile_info: dict[str, Any], keep_grid: np.ndarray, tile_size: int) -> bool:
    tile_position = tile_info.get("tile_position", {})
    tx = tile_position.get("level_x")
    ty = tile_position.get("level_y")
    if tx is None or ty is None:
        gx = int(tile_info.get("gx", 0) or 0)
        gy = int(tile_info.get("gy", 0) or 0)
        tx = gx // max(1, tile_size)
        ty = gy // max(1, tile_size)
    tx = int(tx)
    ty = int(ty)
    if ty < 0 or tx < 0 or ty >= keep_grid.shape[0] or tx >= keep_grid.shape[1]:
        return False
    return bool(keep_grid[ty, tx])


def make_skipped_stain_record(tile_info: dict[str, Any]) -> dict[str, Any]:
    result = {
        "IntensitySumPositive": 0.0,
        "NumberPositive": 0,
        "NumberTotalPixels": int((tile_info.get("width") or 0) * (tile_info.get("height") or 0)),
    }
    return add_tile_metadata(result, tile_info)


def add_tile_metadata(result: dict[str, Any], tile_info: dict[str, Any]) -> dict[str, Any]:
    result["tile_x"] = tile_info.get("x")
    result["tile_y"] = tile_info.get("y")
    result["tile_width"] = tile_info.get("width")
    result["tile_height"] = tile_info.get("height")
    result["tile_level"] = tile_info.get("level")
    result["tile_level_x"] = tile_info.get("level_x")
    result["tile_level_y"] = tile_info.get("level_y")
    result["tile_magnification"] = tile_info.get("magnification")
    result["tile_mm_x"] = tile_info.get("mm_x")
    result["tile_mm_y"] = tile_info.get("mm_y")

    result["tile_gx"] = tile_info.get("gx")
    result["tile_gy"] = tile_info.get("gy")
    result["tile_gwidth"] = tile_info.get("gwidth")
    result["tile_gheight"] = tile_info.get("gheight")

    result["tile_scaled"] = tile_info.get("scaled")
    result["tile_tile_x"] = tile_info.get("tile_x")
    result["tile_tile_y"] = tile_info.get("tile_y")
    result["tile_tile_width"] = tile_info.get("tile_width")
    result["tile_tile_height"] = tile_info.get("tile_height")
    result["tile_tile_magnification"] = tile_info.get("tile_magnification")
    result["tile_tile_mm_x"] = tile_info.get("tile_mm_x")
    result["tile_tile_mm_y"] = tile_info.get("tile_mm_y")

    tile_position = tile_info.get("tile_position", {})
    result["tile_position_level_x"] = tile_position.get("level_x")
    result["tile_position_level_y"] = tile_position.get("level_y")
    result["tile_position_region_x"] = tile_position.get("region_x")
    result["tile_position_region_y"] = tile_position.get("region_y")
    result["tile_position_position"] = tile_position.get("position")

    iterator_range = tile_info.get("iterator_range", {})
    result["iterator_level_x_min"] = iterator_range.get("level_x_min")
    result["iterator_level_y_min"] = iterator_range.get("level_y_min")
    result["iterator_level_x_max"] = iterator_range.get("level_x_max")
    result["iterator_level_y_max"] = iterator_range.get("level_y_max")
    result["iterator_region_x_max"] = iterator_range.get("region_x_max")
    result["iterator_region_y_max"] = iterator_range.get("region_y_max")
    result["iterator_position"] = iterator_range.get("position")

    tile_overlap = tile_info.get("tile_overlap", {})
    result["tile_overlap_left"] = tile_overlap.get("left")
    result["tile_overlap_top"] = tile_overlap.get("top")
    result["tile_overlap_right"] = tile_overlap.get("right")
    result["tile_overlap_bottom"] = tile_overlap.get("bottom")
    return result


def extract_stain(tile_np: np.ndarray, tile_info: dict[str, Any], ppc_params: ppc.Parameters) -> dict[str, Any]:
    stats, _ = ppc.count_image(tile_np, ppc_params)
    return add_tile_metadata(stats._asdict(), tile_info)


def extract_nuclei(
    tile_np: np.ndarray,
    tile_info: dict[str, Any],
    mean_ref: np.ndarray,
    std_ref: np.ndarray,
    stain_matrix: np.ndarray,
) -> dict[str, Any]:
    tile_rgb = maybe_strip_alpha(tile_np)
    im_nmzd = htk.preprocessing.color_normalization.reinhard(tile_rgb, mean_ref, std_ref)
    im_stains = htk.preprocessing.color_deconvolution.color_deconvolution(
        im_nmzd,
        stain_matrix,
    ).Stains
    im_nuclei_stain = im_stains[:, :, 0]
    foreground_threshold = 140
    im_fgnd_mask = sp.ndimage.binary_fill_holes(im_nuclei_stain < foreground_threshold)
    nuclei_mask = morphology.remove_small_objects(im_fgnd_mask, min_size=100)
    nuclei_mask = morphology.remove_small_holes(nuclei_mask, area_threshold=50)
    labeled = measure.label(nuclei_mask)
    nuclei_count = int(labeled.max())

    return add_tile_metadata({"nuclei_count": nuclei_count}, tile_info)


def _process_tile_worker(tile_info: dict[str, Any], run_stain: bool, run_nuclei: bool) -> dict[str, Any]:
    tile_np = tile_info["tile"]
    result: dict[str, Any] = {
        "coord": add_tile_metadata({}, tile_info),
        "stain": None,
        "nuclei": None,
        "errors": {},
        "bands": int(tile_np.shape[2]) if tile_np.ndim == 3 else 1,
    }

    if run_stain:
        try:
            assert _worker_ppc_params is not None
            result["stain"] = extract_stain(tile_np, tile_info, _worker_ppc_params)
        except Exception as exc:
            result["errors"]["stain"] = str(exc)

    if run_nuclei:
        try:
            assert _worker_mean_ref is not None
            assert _worker_std_ref is not None
            assert _worker_stain_matrix is not None
            result["nuclei"] = extract_nuclei(
                tile_np,
                tile_info,
                _worker_mean_ref,
                _worker_std_ref,
                _worker_stain_matrix,
            )
        except Exception as exc:
            result["errors"]["nuclei"] = str(exc)

    return result


def build_grid_index(records: list[dict[str, Any]]) -> tuple[list[int], list[int], dict[int, int], dict[int, int]]:
    gx_vals = sorted({int(r["tile_gx"]) for r in records if r.get("tile_gx") is not None})
    gy_vals = sorted({int(r["tile_gy"]) for r in records if r.get("tile_gy") is not None})
    return gx_vals, gy_vals, {gx: i for i, gx in enumerate(gx_vals)}, {gy: i for i, gy in enumerate(gy_vals)}


def build_metric_arrays(tiled_info: pd.DataFrame, metrics: list[str]) -> dict[str, np.ndarray]:
    required = {"tile_gx", "tile_gy"}
    if not required.issubset(tiled_info.columns):
        return {}

    gx_vals = sorted({int(v) for v in tiled_info["tile_gx"].dropna().astype(int)})
    gy_vals = sorted({int(v) for v in tiled_info["tile_gy"].dropna().astype(int)})
    if not gx_vals or not gy_vals:
        return {}

    x_index = {gx: i for i, gx in enumerate(gx_vals)}
    y_index = {gy: i for i, gy in enumerate(gy_vals)}
    arrays: dict[str, np.ndarray] = {}
    for metric in metrics:
        if metric not in tiled_info.columns:
            continue
        grid = np.full((len(gy_vals), len(gx_vals)), np.nan, dtype=float)
        for _, row in tiled_info.iterrows():
            gx = row.get("tile_gx")
            gy = row.get("tile_gy")
            if pd.isna(gx) or pd.isna(gy):
                continue
            ix = x_index.get(int(gx))
            iy = y_index.get(int(gy))
            if ix is None or iy is None:
                continue
            val = row.get(metric)
            try:
                grid[iy, ix] = float(val) if np.isfinite(val) else np.nan
            except Exception:
                grid[iy, ix] = np.nan
        arrays[metric] = grid
    return arrays


def save_metric_arrays(arrays: dict[str, np.ndarray], output_dir: Path) -> None:
    for metric, arr in arrays.items():
        np.save(output_dir / f"{metric}_map.npy", arr)


def save_stain_preview(
    slide_path: Path,
    arrays: dict[str, np.ndarray],
    output_dir: Path,
    *,
    metric_preference: tuple[str, ...] = ("PercentagePositive",),
) -> dict[str, Any] | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_name = next((name for name in metric_preference if name in arrays), None)
    if metric_name is None:
        return None

    arr = np.nan_to_num(arrays[metric_name].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    slide = openslide.OpenSlide(str(slide_path))
    thumbnail = np.array(slide.associated_images["thumbnail"])
    th_h, th_w = thumbnail.shape[:2]

    resized = cv2.resize(arr, (th_w, th_h), interpolation=cv2.INTER_NEAREST)
    padded, transform = fit_and_pad_2d(resized.astype(np.float32), (1024, 1024), interpolation=cv2.INTER_NEAREST)
    padded = padded.astype(np.float32)
    lo = 0.0
    hi = 0.02
    norm = (padded - lo) / (hi - lo)
    scaling = {"mode": "fixed", "min": lo, "max": hi, "units": "fraction_positive"}
    norm = np.clip(norm, 0.0, 1.0)
    preview_u8 = (norm * 255.0).astype(np.uint8)
    preview_path = output_dir / f"{slide_path.stem}_0002.tif"
    cv2.imwrite(str(preview_path), preview_u8)
    return {
        "kind": "stain",
        "metric_name": metric_name,
        "map_height": int(arr.shape[0]),
        "map_width": int(arr.shape[1]),
        "thumbnail_height": int(th_h),
        "thumbnail_width": int(th_w),
        "preview_tif": str(preview_path),
        "transform": transform,
        "scaling": scaling,
        "reverse_mapping": {
            "prediction_crop": {
                "y0": transform["crop_y0"],
                "y1": transform["crop_y1"],
                "x0": transform["crop_x0"],
                "x1": transform["crop_x1"],
            },
            "resize_prediction_to_map": {
                "width": int(arr.shape[1]),
                "height": int(arr.shape[0]),
                "interpolation": "nearest",
            },
        },
    }


def build_nuclei_density_map(records: list[dict[str, Any]]) -> tuple[np.ndarray, int, int]:
    gx_vals, gy_vals, x_index, y_index = build_grid_index(records)
    density_map = np.zeros((len(gy_vals), len(gx_vals)), dtype=np.float32)
    for record in records:
        gx = record.get("tile_gx")
        gy = record.get("tile_gy")
        if gx is None or gy is None:
            continue
        density_map[y_index[int(gy)], x_index[int(gx)]] = float(record.get("nuclei_count", 0.0))
    return density_map, len(gx_vals), len(gy_vals)


def save_nuclei_outputs(
    slide_path: Path,
    density_map: np.ndarray,
    output_dir: Path,
    reference_image: str,
    bands: int,
    tile_size: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = slide_path.stem
    np.save(output_dir / "nuclei_density_map.npy", density_map)

    slide = openslide.OpenSlide(str(slide_path))
    thumbnail = np.array(slide.associated_images["thumbnail"])
    th_h, th_w = thumbnail.shape[:2]

    resized_counts = cv2.resize(density_map, (th_w, th_h), interpolation=cv2.INTER_NEAREST)
    padded, transform = fit_and_pad_2d(resized_counts.astype(np.float32), (1024, 1024), interpolation=cv2.INTER_NEAREST)
    to_save = np.clip(padded, 0, 255).astype(np.uint8)
    preview_path = output_dir / f"{stem}_0001.tif"
    cv2.imwrite(str(preview_path), to_save)
    return {
        "kind": "nuclei",
        "map_height": int(density_map.shape[0]),
        "map_width": int(density_map.shape[1]),
        "thumbnail_height": int(th_h),
        "thumbnail_width": int(th_w),
        "preview_tif": str(preview_path),
        "transform": transform,
        "reference_image": reference_image,
        "normalization": "reinhard",
        "bands": int(bands),
        "tile_size": int(tile_size),
        "reverse_mapping": {
            "prediction_crop": {
                "y0": transform["crop_y0"],
                "y1": transform["crop_y1"],
                "x0": transform["crop_x0"],
                "x1": transform["crop_x1"],
            },
            "resize_prediction_to_map": {
                "width": int(density_map.shape[1]),
                "height": int(density_map.shape[0]),
                "interpolation": "nearest",
            },
        },
    }


def write_sample_metadata(
    sample_output_dir: Path,
    slide_path: Path,
    tile_size: int,
    bands: int,
    clahe_meta: dict[str, Any] | None,
    stain_meta: dict[str, Any] | None,
    nuclei_meta: dict[str, Any] | None,
) -> None:
    sample_output_dir.mkdir(parents=True, exist_ok=True)
    slide = openslide.OpenSlide(str(slide_path))
    thumbnail = np.array(slide.associated_images["thumbnail"])
    th_h, th_w = thumbnail.shape[:2]
    metadata = {
        "slide_path": str(slide_path.resolve()),
        "tile_size": int(tile_size),
        "slide_width": int(slide.dimensions[0]),
        "slide_height": int(slide.dimensions[1]),
        "bands": int(bands),
        "thumbnail_width": int(th_w),
        "thumbnail_height": int(th_h),
        "transforms": {},
        "maps": {},
    }
    if clahe_meta is not None:
        metadata["transforms"]["clahe"] = clahe_meta
    if stain_meta is not None:
        metadata["transforms"]["stain"] = stain_meta
        metadata["maps"]["stain"] = {
            "metric_name": stain_meta["metric_name"],
            "map_width": stain_meta["map_width"],
            "map_height": stain_meta["map_height"],
        }
    if nuclei_meta is not None:
        metadata["transforms"]["nuclei"] = nuclei_meta
        metadata["maps"]["nuclei"] = {
            "map_width": nuclei_meta["map_width"],
            "map_height": nuclei_meta["map_height"],
            "reference_image": nuclei_meta["reference_image"],
            "normalization": nuclei_meta["normalization"],
        }
    with open(sample_output_dir / "meta.json", "w") as f:
        json.dump(metadata, f, indent=2)


def save_clahe_output(
    slide_path: Path,
    output_dir: Path,
    pad: int = 1024,
    gamma: float = 0.5,
    clahe_clip: float = 12.0,
    clahe_grid: int = 4,
    thresh: int = 13,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    slide = openslide.OpenSlide(str(slide_path))
    thumb = np.array(slide.associated_images["thumbnail"])
    img_gray = cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY)
    labeled, _ = get_tissue_mask(255 - img_gray)
    mask = labeled > 0
    tissue_only = np.where(mask, img_gray, 0).astype(np.uint8)
    padded, transform = fit_and_pad_2d(tissue_only, (pad, pad), interpolation=cv2.INTER_NEAREST, pad_value=0)
    padded = padded.astype(np.uint8)
    gamma_corrected = apply_gamma(padded, gamma=gamma)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
    enhanced = clahe.apply(gamma_corrected)
    enhanced[enhanced <= thresh] = 0
    preview_path = output_dir / f"{slide_path.stem}_0000.tif"
    cv2.imwrite(str(preview_path), enhanced)
    return {
        "kind": "clahe",
        "thumbnail_height": int(thumb.shape[0]),
        "thumbnail_width": int(thumb.shape[1]),
        "preview_tif": str(preview_path),
        "transform": transform,
        "gamma": float(gamma),
        "clahe_clip": float(clahe_clip),
        "clahe_grid": int(clahe_grid),
        "threshold": int(thresh),
    }


@dataclass
class SlideStatus:
    enabled: list[str]
    success: dict[str, bool] = field(default_factory=dict)
    errors: dict[str, list[str]] = field(default_factory=dict)
    tile_failures: dict[str, int] = field(default_factory=dict)


class UnifiedSlideProcessor:
    def __init__(
        self,
        slide_path: str | Path,
        output_root: str | Path,
        *,
        run_clahe: bool = True,
        run_stain: bool = True,
        run_nuclei: bool = True,
        magnification: int = 40,
        tile_size: int = 512,
        workers: int | None = None,
        reference_image: str = DEFAULT_REFERENCE_IMAGE,
    ) -> None:
        self.slide_path = Path(slide_path).resolve()
        self.output_root = Path(output_root).resolve()
        self.run_clahe_stage = run_clahe
        self.run_stain_stage = run_stain
        self.run_nuclei_stage = run_nuclei
        self.magnification = magnification
        self.tile_size = tile_size
        self.workers = workers or max(1, min(os.cpu_count() or 1, 8))
        self.reference_image = reference_image
        self.template_params = {
            "magnification": self.magnification,
            "tile_size": {"width": self.tile_size, "height": self.tile_size},
            "tile_overlap": {"x": 0, "y": 0},
            "positive_pixel_count": {
                "hue_value": 0.05,
                "hue_width": 0.15,
                "saturation_minimum": 0.05,
                "intensity_upper_limit": 0.95,
                "intensity_weak_threshold": 0.65,
                "intensity_strong_threshold": 0.35,
                "intensity_lower_limit": 0.05,
            },
        }

    @property
    def enabled_stages(self) -> list[str]:
        stages = []
        if self.run_clahe_stage:
            stages.append("clahe")
        if self.run_stain_stage:
            stages.append("stain")
        if self.run_nuclei_stage:
            stages.append("nuclei")
        return stages

    def _load_nuclei_reference(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ref_path = Path(self.reference_image)
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference image not found: {ref_path}")
        im_reference = np.load(ref_path)
        mean_ref, std_ref = htk.preprocessing.color_conversion.lab_mean_std(im_reference)
        stain_color_map = {
            "hematoxylin": [0.65, 0.70, 0.29],
            "dab": [0.27, 0.57, 0.78],
            "null": [0.0, 0.0, 0.0],
        }
        stain_matrix = np.array(
            [
                stain_color_map["hematoxylin"],
                stain_color_map["dab"],
                stain_color_map["null"],
            ]
        ).T
        return mean_ref, std_ref, stain_matrix

    def _run_tile_pipeline(self, status: SlideStatus) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
        ts = large_image.getTileSource(str(self.slide_path))
        md = ts.getMetadata()
        slide_size = (int(md.get("sizeX", 0)), int(md.get("sizeY", 0)))
        tile_iterator = ts.tileIterator(
            scale=dict(magnification=self.magnification),
            tile_size=self.template_params["tile_size"],
            tile_overlap=self.template_params["tile_overlap"],
            format=large_image.tilesource.TILE_FORMAT_NUMPY,
        )
        ppc_params_dict = self.template_params["positive_pixel_count"]
        mean_ref = std_ref = stain_matrix = None
        if self.run_nuclei_stage:
            mean_ref, std_ref, stain_matrix = self._load_nuclei_reference()

        stain_records: list[dict[str, Any]] = []
        nuclei_records: list[dict[str, Any]] = []
        first_bands: int | None = None
        processed = 0
        skipped = 0
        started_at = time.monotonic()
        total_tiles: int | None = None
        tissue_overlap_threshold = 0.0
        tissue_mask, thumb_size = build_thumbnail_tissue_mask(self.slide_path)
        keep_grid = build_tile_keep_grid(
            tissue_mask,
            thumb_size,
            slide_size,
            self.tile_size,
            overlap_threshold=tissue_overlap_threshold,
            dilate_tiles=1,
        )
        tissue_coverage = float(tissue_mask.mean()) if tissue_mask.size else 0.0
        kept_tiles = int(keep_grid.sum())
        total_grid_tiles = int(keep_grid.size)

        log(
            f"Starting tile pass for {self.slide_path.name}: "
            f"stages={','.join(self.enabled_stages)} "
            f"magnification={self.magnification} tile_size={self.tile_size} workers={self.workers}"
        )
        log(
            f"Thumbnail tissue mask ready: size={thumb_size[0]}x{thumb_size[1]} "
            f"coverage={tissue_coverage*100:.2f}% overlap_threshold={tissue_overlap_threshold:.4f}"
        )
        log(f"Tile keep grid ready: kept={kept_tiles}/{total_grid_tiles} tiles")

        with cf.ProcessPoolExecutor(
            max_workers=self.workers,
            initializer=_worker_init,
            initargs=(ppc_params_dict if self.run_stain_stage else None, mean_ref, std_ref, stain_matrix),
        ) as executor:
            pending: set[cf.Future] = set()
            iterator = iter(tile_iterator)
            max_pending = max(1, self.workers * 2)

            def submit_one(tile_info: dict[str, Any]) -> bool:
                nonlocal processed, skipped, first_bands
                if not should_keep_tile(tile_info, keep_grid, self.tile_size):
                    coord = add_tile_metadata({}, tile_info)
                    if self.run_stain_stage:
                        stain_records.append(make_skipped_stain_record(tile_info))
                    if self.run_nuclei_stage:
                        nuclei_records.append(dict(coord, nuclei_count=0))
                    if first_bands is None:
                        tile_np = tile_info["tile"]
                        first_bands = int(tile_np.shape[2]) if tile_np.ndim == 3 else 1
                    processed += 1
                    skipped += 1
                    return False
                pending.add(
                    executor.submit(
                        _process_tile_worker,
                        tile_info,
                        self.run_stain_stage,
                        self.run_nuclei_stage,
                    )
                )
                return True

            iterator_exhausted = False

            def fill_pending() -> None:
                nonlocal iterator_exhausted, total_tiles
                while not iterator_exhausted and len(pending) < max_pending:
                    try:
                        tile_info = next(iterator)
                        if total_tiles is None:
                            total_tiles = tile_info.get("iterator_range", {}).get("position")
                        submit_one(tile_info)
                    except StopIteration:
                        iterator_exhausted = True
                        break

            fill_pending()

            if total_tiles:
                log(f"Total tiles scheduled: {total_tiles}")
            else:
                log("Total tile count unavailable; reporting processed tiles only")

            progress_every = 100
            if total_tiles:
                progress_every = max(100, int(total_tiles) // 20)

            while pending or not iterator_exhausted:
                if not pending:
                    fill_pending()
                    if not pending and iterator_exhausted:
                        break
                future = next(cf.as_completed(pending))
                pending.remove(future)
                tile_result = future.result()
                processed += 1
                coord = tile_result["coord"]
                if first_bands is None:
                    first_bands = tile_result["bands"]
                for stage_name, err in tile_result["errors"].items():
                    status.tile_failures[stage_name] = status.tile_failures.get(stage_name, 0) + 1
                    status.errors.setdefault(stage_name, []).append(err)
                if self.run_stain_stage:
                    stain_records.append(tile_result["stain"] or dict(coord))
                if self.run_nuclei_stage:
                    nuclei_record = tile_result["nuclei"] or dict(coord, nuclei_count=0)
                    nuclei_records.append(nuclei_record)
                fill_pending()

                if processed == 1 or processed % progress_every == 0 or (total_tiles and processed == total_tiles):
                    elapsed = time.monotonic() - started_at
                    rate = processed / elapsed if elapsed > 0 else 0.0
                    if total_tiles:
                        pct = (processed / total_tiles) * 100.0
                        remaining_tiles = max(0, total_tiles - processed)
                        eta_seconds = remaining_tiles / rate if rate > 0 else 0.0
                        log(
                            f"Progress: {processed}/{total_tiles} tiles "
                            f"({pct:.1f}%) skipped={skipped} elapsed={elapsed/60:.1f}m eta={eta_seconds/60:.1f}m"
                        )
                    else:
                        log(
                            f"Progress: {processed} tiles processed "
                            f"skipped={skipped} elapsed={elapsed/60:.1f}m rate={rate:.2f} tiles/s"
                        )

        elapsed = time.monotonic() - started_at
        child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu_user = float(child_usage.ru_utime)
        cpu_sys = float(child_usage.ru_stime)
        cpu_total = cpu_user + cpu_sys
        denom = elapsed * max(1, self.workers)
        efficiency = (cpu_total / denom) if denom > 0 else 0.0

        log(f"Tile pass complete: processed={processed} skipped={skipped} elapsed={elapsed/60:.1f}m")
        log(f"Tile wall time: {elapsed:.1f}s")
        log(f"Tile CPU time (children user): {cpu_user:.1f}s")
        log(f"Tile CPU time (children total): {cpu_total:.1f}s")
        log(f"Parallelism efficiency: {efficiency:.2f}")

        return stain_records, nuclei_records, first_bands or 3

    def run(self) -> SlideStatus:
        status = SlideStatus(enabled=self.enabled_stages)
        if not self.slide_path.exists():
            raise FileNotFoundError(f"Slide not found: {self.slide_path}")
        if not self.enabled_stages:
            raise ValueError("At least one stage must be enabled")

        self.output_root.mkdir(parents=True, exist_ok=True)
        clahe_meta: dict[str, Any] | None = None
        stain_meta: dict[str, Any] | None = None
        nuclei_meta: dict[str, Any] | None = None
        bands_for_meta = 3

        if self.run_clahe_stage:
            try:
                log(f"Starting CLAHE for {self.slide_path.name}")
                clahe_meta = save_clahe_output(self.slide_path, self.output_root / "clahe")
                log(f"Finished CLAHE for {self.slide_path.name}")
                status.success["clahe"] = True
            except Exception as exc:
                status.success["clahe"] = False
                status.errors.setdefault("clahe", []).append(str(exc))

        if self.run_stain_stage or self.run_nuclei_stage:
            try:
                stain_records, nuclei_records, bands = self._run_tile_pipeline(status)
                bands_for_meta = int(bands)

                if self.run_stain_stage:
                    stain_df = pd.DataFrame(stain_records)
                    if (
                        len(stain_df) > 0
                        and "NumberPositive" in stain_df.columns
                        and "NumberTotalPixels" in stain_df.columns
                    ):
                        denom = stain_df["NumberTotalPixels"].replace(0, np.nan)
                        stain_df["PercentagePositive"] = stain_df["NumberPositive"] / denom
                    stain_arrays = build_metric_arrays(
                        stain_df,
                        ["IntensitySumPositive", "PercentagePositive", "NumberPositive"],
                    )
                    stain_out = self.output_root / "stain"
                    stain_out.mkdir(parents=True, exist_ok=True)
                    save_metric_arrays(stain_arrays, stain_out)
                    stain_meta = save_stain_preview(self.slide_path, stain_arrays, stain_out)
                    status.success["stain"] = True

                if self.run_nuclei_stage:
                    density_map, _, _ = build_nuclei_density_map(nuclei_records)
                    nuclei_out = self.output_root / "nuclei"
                    nuclei_meta = save_nuclei_outputs(
                        self.slide_path,
                        density_map,
                        nuclei_out,
                        self.reference_image,
                        bands,
                        self.tile_size,
                    )
                    status.success["nuclei"] = True
            except Exception as exc:
                if self.run_stain_stage:
                    status.success.setdefault("stain", False)
                    status.errors.setdefault("stain", []).append(str(exc))
                if self.run_nuclei_stage:
                    status.success.setdefault("nuclei", False)
                    status.errors.setdefault("nuclei", []).append(str(exc))

        if self.run_clahe_stage or self.run_stain_stage or self.run_nuclei_stage:
            try:
                write_sample_metadata(
                    self.output_root,
                    self.slide_path,
                    self.tile_size,
                    bands_for_meta,
                    clahe_meta,
                    stain_meta,
                    nuclei_meta,
                )
            except Exception as exc:
                status.errors.setdefault("meta", []).append(str(exc))

        return status


def find_wsi_files(images_dir: str | Path) -> list[Path]:
    images_path = Path(images_dir).resolve()
    matches: set[Path] = set()
    for ext in WSI_EXTS:
        matches.update(images_path.glob(f"*{ext}"))
        matches.update(images_path.glob(f"*{ext.upper()}"))
    return sorted(matches)
