#!/usr/bin/env python3
"""Inspect a run directory without looking at the images.

Why this exists
---------------
A capture can "succeed" and still be useless: an all-black frame, a frame that
is one flat colour, or a picture taken from inside a wall. Pixel statistics
catch those cases without a human (or a vision model) looking at the picture.

It also summarises a scene profile's asset list, which is what decides whether
building parts are available on a given scene.

Usage
-----
    python scripts/inspect_run.py --run runs/loop1
    python scripts/inspect_run.py --run runs/loop1 \
        --profile runs/smoke_openuav/profile_openuav_moderncity.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

try:
    from PIL import Image
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"needs Pillow and numpy: {exc}")


def image_stats(path: Path) -> Dict[str, object]:
    """Basic statistics that reveal blank, black or flat captures."""
    with Image.open(path) as handle:
        rgb = handle.convert("RGB")
        array = np.asarray(rgb, dtype=np.float32)

    mean = array.mean(axis=(0, 1))
    overall = float(mean.mean())
    std = float(array.std())
    dark = float((array.max(axis=2) < 12).mean())
    bright = float((array.min(axis=2) > 243).mean())

    # How many distinct colours, quantised coarsely -- a flat frame has very few.
    quantised = (array // 16).astype(np.uint8).reshape(-1, 3)
    unique = len({tuple(row) for row in quantised[::7]})

    return {
        "size": f"{rgb.width}x{rgb.height}",
        "mean_rgb": tuple(round(float(v), 1) for v in mean),
        "brightness": round(overall, 1),
        "std": round(std, 1),
        "dark_pct": round(dark * 100, 1),
        "white_pct": round(bright * 100, 1),
        "colour_buckets": unique,
    }


def verdict(stats: Dict[str, object]) -> str:
    """Turn statistics into a plain-language judgement."""
    if stats["dark_pct"] > 95:
        return "BLACK -- nothing rendered"
    if stats["white_pct"] > 95:
        return "WHITE -- overexposed or empty sky"
    if stats["std"] < 3:
        return "FLAT -- one solid colour, probably inside geometry"
    if stats["colour_buckets"] < 6:
        return "LOW VARIETY -- may be a blank surface"
    return "ok"


def summarise_assets(profile_path: Path, top: int = 25) -> None:
    """Group the asset list by name prefix and surface building-related names."""
    data = json.loads(profile_path.read_text(encoding="utf-8-sig"))
    assets: List[str] = [str(a) for a in data.get("assets", [])]

    print()
    print("=" * 78)
    print(f"scene profile: {profile_path.name}")
    print("=" * 78)
    print(f"  scene_id : {data.get('scene_id')}")
    print(f"  port     : {data.get('port')}")
    print(f"  vehicle  : {data.get('vehicle')}")
    print(f"  cameras  : {data.get('cameras')}")
    print(f"  assets   : {len(assets)}")

    buckets: Dict[str, int] = {}
    for name in assets:
        prefix = name.split("_")[0] if "_" in name else name
        buckets[prefix] = buckets.get(prefix, 0) + 1

    print()
    print("  by prefix:")
    for prefix, count in sorted(buckets.items(), key=lambda kv: -kv[1])[:top]:
        print(f"    {prefix:<14} {count}")

    keywords = ("wall", "window", "door", "floor", "roof", "build", "house",
                "facade", "stairs", "pillar", "column", "room", "apartment")
    relevant = [a for a in assets if any(k in a.lower() for k in keywords)]
    print()
    print(f"  building-related names: {len(relevant)}")
    for name in relevant[:40]:
        print("    ", name)

    people = [a for a in assets if a.startswith("BP_") or "char" in a.lower()]
    print()
    print(f"  person-like names: {len(people)}")
    for name in people[:20]:
        print("    ", name)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarise a render run and a scene profile",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run", type=Path, default=Path("runs/loop1"))
    parser.add_argument("--profile", type=Path, default=None)
    args = parser.parse_args(argv)

    images = sorted(args.run.rglob("*.png")) if args.run.is_dir() else []
    if images:
        print("=" * 78)
        print(f"images in {args.run}: {len(images)}")
        print("=" * 78)
        print(f"  {'file':<42} {'size':>9} {'bright':>7} {'std':>6} "
              f"{'dark%':>6} {'white%':>7} {'buckets':>7}  verdict")
        print("  " + "-" * 110)
        for path in images:
            stats = image_stats(path)
            rel = path.relative_to(args.run)
            print(
                f"  {str(rel):<42} {str(stats['size']):>9} "
                f"{stats['brightness']:>7} {stats['std']:>6} "
                f"{stats['dark_pct']:>6} {stats['white_pct']:>7} "
                f"{stats['colour_buckets']:>7}  {verdict(stats)}"
            )
    else:
        print(f"no images found under {args.run}")

    if args.profile:
        summarise_assets(args.profile)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())