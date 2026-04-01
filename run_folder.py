#!/usr/bin/env python3
"""Run unified pipeline for every WSI in a folder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))


def resolve_stage_flags(args: argparse.Namespace) -> tuple[bool, bool, bool]:
    any_flag = args.clahe or args.stain or args.nuclei
    if args.all or not any_flag:
        return True, True, True
    return args.clahe, args.stain, args.nuclei


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified histology pipeline for a folder of slides")
    parser.add_argument("images_dir", help="Directory containing WSI slides")
    parser.add_argument("output_dir", help="Output root directory")
    parser.add_argument("--clahe", action="store_true", help="Run CLAHE stage")
    parser.add_argument("--stain", action="store_true", help="Run stain stage")
    parser.add_argument("--nuclei", action="store_true", help="Run nuclei stage")
    parser.add_argument("--all", action="store_true", help="Run all stages (default)")
    parser.add_argument("--magnification", type=int, default=40, help="Tile magnification")
    parser.add_argument("--tile-size", type=int, default=512, help="Tile size")
    parser.add_argument("--workers", type=int, default=None, help="Worker threads")
    parser.add_argument("--force", action="store_true", help="Reprocess existing slide output folders")
    parser.add_argument(
        "--reference-image",
        default=None,
        help="Reference tile .npy used for nuclei normalization",
    )
    args = parser.parse_args()

    if __package__ in {None, ""}:
        from pipeline import UnifiedSlideProcessor, find_wsi_files
    else:
        from .pipeline import UnifiedSlideProcessor, find_wsi_files

    run_clahe, run_stain, run_nuclei = resolve_stage_flags(args)
    slides = find_wsi_files(args.images_dir)
    if not slides:
        print(f"No WSI files found in {Path(args.images_dir).resolve()}", file=sys.stderr)
        return 0

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    exit_code = 0

    for slide in slides:
        slide_root = output_dir / slide.stem
        if slide_root.exists() and not args.force:
            print(f"Skipping {slide.stem}: output exists at {slide_root}")
            continue
        kwargs = {}
        if args.reference_image:
            kwargs["reference_image"] = args.reference_image
        processor = UnifiedSlideProcessor(
            slide,
            slide_root,
            run_clahe=run_clahe,
            run_stain=run_stain,
            run_nuclei=run_nuclei,
            magnification=args.magnification,
            tile_size=args.tile_size,
            workers=args.workers,
            **kwargs,
        )
        status = processor.run()
        results[slide.stem] = status.__dict__
        if not all(status.success.get(stage, False) for stage in status.enabled):
            exit_code = 1

    print(json.dumps(results, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
