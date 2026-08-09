"""Map Stretch's aggregate arm actuator through its ROS virtual-joint model."""

from __future__ import annotations

from typing import Sequence

import numpy as np

STRETCH_EXTERNAL_DIMENSION = 11
STRETCH_MIMIC_DIMENSION = 8
ARM_SEGMENT_SLICE = slice(4, 8)


def reduce_stretch_mimic_vector(values: Sequence[float]) -> np.ndarray:
    """Reduce four virtual URDF joints to q_arm = wrist_extension / 4."""

    external = _finite_vector(values, STRETCH_EXTERNAL_DIMENSION, "external")
    return np.concatenate(
        (
            external[:4],
            [float(np.mean(external[ARM_SEGMENT_SLICE]))],
            external[8:],
        )
    )


def expand_stretch_mimic_vector(values: Sequence[float]) -> np.ndarray:
    """Expand q_arm to four ROS geometry joints; these are not four actuators."""

    internal = _finite_vector(values, STRETCH_MIMIC_DIMENSION, "mimic")
    return np.concatenate((internal[:4], np.repeat(internal[4], 4), internal[5:]))


def expand_stretch_mimic_state(values: Sequence[float]) -> np.ndarray:
    """Expand an 8-q/8-v state or state-limit vector to 11-q/11-v."""

    state = _finite_vector(values, 2 * STRETCH_MIMIC_DIMENSION, "mimic state")
    return np.concatenate(
        (
            expand_stretch_mimic_vector(state[:STRETCH_MIMIC_DIMENSION]),
            expand_stretch_mimic_vector(state[STRETCH_MIMIC_DIMENSION:]),
        )
    )


def _finite_vector(values: Sequence[float], size: int, label: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.size != size:
        raise ValueError(f"{label} vector must have length {size}, got {vector.size}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} vector contains NaN or Inf")
    return vector.copy()
