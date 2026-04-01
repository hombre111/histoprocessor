#!/usr/bin/env python3
"""Run unified pipeline for a single slide."""

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
    parser = argparse.ArgumentParser(description="Unified histology pipeline for a single slide")
    parser.add_argument("slide_path", help="Path to a single WSI slide")
    parser.add_argument("output_root", help="Output root for this slide")
    parser.add_argument("--clahe", action="store_true", help="Run CLAHE stage")
    parser.add_argument("--stain", action="store_true", help="Run stain stage")
    parser.add_argument("--nuclei", action="store_true", help="Run nuclei stage")
    parser.add_argument("--all", action="store_true", help="Run all stages (default)")
    parser.add_argument("--magnification", type=int, default=40, help="Tile magnification")
    parser.add_argument("--tile-size", type=int, default=512, help="Tile size")
    parser.add_argument("--workers", type=int, default=None, help="Worker threads")
    parser.add_argument(
        "--reference-image",
        default=None,
        help="Reference tile .npy used for nuclei normalization",
    )
    args = parser.parse_args()

    if __package__ in {None, ""}:
        from pipeline import UnifiedSlideProcessor
    else:
        from .pipeline import UnifiedSlideProcessor

    run_clahe, run_stain, run_nuclei = resolve_stage_flags(args)
    kwargs = {}
    if args.reference_image:
        kwargs["reference_image"] = args.reference_image

    processor = UnifiedSlideProcessor(
        args.slide_path,
        args.output_root,
        run_clahe=run_clahe,
        run_stain=run_stain,
        run_nuclei=run_nuclei,
        magnification=args.magnification,
        tile_size=args.tile_size,
        workers=args.workers,
        **kwargs,
    )
    status = processor.run()
    print(json.dumps(status.__dict__, indent=2))
    return 0 if all(status.success.get(stage, False) for stage in status.enabled) else 1


if __name__ == "__main__":
    raise SystemExit(main())
