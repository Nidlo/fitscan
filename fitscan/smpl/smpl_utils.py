"""SMPL utility functions. Scale, joint helpers."""

from __future__ import annotations

import numpy as np


def scale_mesh_to_height(vertices: np.ndarray, target_height_cm: float) -> np.ndarray:
    """Scale SMPL mesh vertices so total height matches target."""
    y_min = vertices[:, 1].min()
    y_max = vertices[:, 1].max()
    current_height_m = y_max - y_min
    if current_height_m <= 0:
        return vertices
    target_height_m = target_height_cm / 100.0
    scale = target_height_m / current_height_m
    return vertices * scale
