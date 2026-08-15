#!/usr/bin/env python3
"""Fill a tightly bounded ESDF unknown hole.

By default, unknown voxels copy the nearest observed ESDF sample.  A constant
positive distance can instead be used for a physically verified free-space
region, such as an obstacle-free slab above a real-robot work area.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def _axis_indices(axis: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return np.flatnonzero((axis >= lower) & (axis <= upper))


def patch(args) -> dict:
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if source == output:
        raise ValueError("output must differ from input; the source map is preserved")

    with np.load(source) as loaded:
        arrays = {name: np.asarray(loaded[name]).copy() for name in loaded.files}

    xs = np.asarray(arrays["xs"], dtype=np.float64)
    ys = np.asarray(arrays["ys"], dtype=np.float64)
    zs = np.asarray(arrays["zs"], dtype=np.float64)
    distance = arrays["distance"]
    gradient = arrays["gradient"]
    valid = arrays["valid"]

    xmin, xmax, ymin, ymax, zmin, zmax = args.bounds
    xi = _axis_indices(xs, xmin, xmax)
    yi = _axis_indices(ys, ymin, ymax)
    zi = _axis_indices(zs, zmin, zmax)
    if min(len(xi), len(yi), len(zi)) == 0:
        raise ValueError("patch bounds do not overlap the ESDF grid")

    local_valid = valid[np.ix_(xi, yi, zi)]
    local_targets = np.argwhere(~local_valid)
    if len(local_targets) == 0:
        raise ValueError("patch bounds contain no unknown voxels")
    targets = np.column_stack(
        (
            xi[local_targets[:, 0]],
            yi[local_targets[:, 1]],
            zi[local_targets[:, 2]],
        )
    )

    target_key = tuple(targets.T)
    nearest_distance_max = None
    if args.constant_distance is not None:
        fill_distance = float(args.constant_distance)
        if not np.isfinite(fill_distance) or fill_distance <= 0.0:
            raise ValueError("constant distance must be finite and positive")
        distance[target_key] = fill_distance
        gradient[target_key] = 0.0
        fill_mode = "constant_free_distance"
    else:
        padding = float(args.source_padding)
        cxi = _axis_indices(xs, xmin - padding, xmax + padding)
        cyi = _axis_indices(ys, ymin - padding, ymax + padding)
        czi = _axis_indices(zs, zmin - padding, zmax + padding)
        candidate_valid = valid[np.ix_(cxi, cyi, czi)]
        local_candidates = np.argwhere(candidate_valid)
        if len(local_candidates) == 0:
            raise ValueError("no observed source voxels exist near the patch bounds")
        candidates = np.column_stack(
            (
                cxi[local_candidates[:, 0]],
                cyi[local_candidates[:, 1]],
                czi[local_candidates[:, 2]],
            )
        )

        target_xyz = np.column_stack(
            (xs[targets[:, 0]], ys[targets[:, 1]], zs[targets[:, 2]])
        )
        candidate_xyz = np.column_stack(
            (
                xs[candidates[:, 0]],
                ys[candidates[:, 1]],
                zs[candidates[:, 2]],
            )
        )
        nearest_distance, nearest_index = cKDTree(candidate_xyz).query(target_xyz, k=1)
        nearest_distance_max = float(np.max(nearest_distance))
        if nearest_distance_max > float(args.max_source_distance):
            raise ValueError(
                "nearest observed source voxel is too far away: "
                f"{nearest_distance_max:.3f} m"
            )
        nearest = candidates[np.asarray(nearest_index, dtype=np.int64)]
        source_key = tuple(nearest.T)
        distance[target_key] = distance[source_key]
        gradient[target_key] = gradient[source_key]
        fill_mode = "nearest_observed"
    valid[target_key] = True

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(output)
    return {
        "input": str(source),
        "output": str(output),
        "fill_mode": fill_mode,
        "filled_voxels": int(len(targets)),
        "nearest_source_distance_max_m": nearest_distance_max,
        "filled_distance_min_m": float(np.min(distance[target_key])),
        "filled_distance_max_m": float(np.max(distance[target_key])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument(
        "--bounds",
        nargs=6,
        type=float,
        required=True,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
    )
    parser.add_argument("--source-padding", type=float, default=0.20)
    parser.add_argument("--max-source-distance", type=float, default=0.20)
    parser.add_argument(
        "--constant-distance",
        type=float,
        help=(
            "Mark bounded unknown voxels as verified free space with this "
            "constant positive ESDF distance and a zero gradient."
        ),
    )
    args = parser.parse_args()
    print(patch(args))


if __name__ == "__main__":
    main()
