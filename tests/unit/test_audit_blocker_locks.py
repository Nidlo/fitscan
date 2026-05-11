"""Regression locks for audit findings FS-C6 (broken /measure/avatar)
and FS-H13 (mesh never scaled to user height).

Both bugs were already fixed before this sprint; these tests stop a
future refactor from silently re-introducing them. C6 caught us once;
the absence of an integration test is exactly how it shipped.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from fitscan.application.measurement_service import MeasurementService
from fitscan.application.pipeline_orchestrator import (
    PipelineOrchestrator,
    PipelineResult,
)
from fitscan.domain.models import ScaleFactor


class TestC6MeasureWithMeshContract:
    """The /measure/avatar route calls `service.measure_with_mesh(...)`.
    The audit caught this method missing. Lock the signature so a future
    rename can't break the route again."""

    def test_measure_with_mesh_method_exists(self) -> None:
        assert hasattr(MeasurementService, "measure_with_mesh")
        assert callable(MeasurementService.measure_with_mesh)

    def test_measure_with_mesh_signature_is_stable(self) -> None:
        """The avatar route passes these kwargs by name. If the signature
        changes silently, the route 500s — exactly the C6 bug."""
        sig = inspect.signature(MeasurementService.measure_with_mesh)
        param_names = list(sig.parameters.keys())
        # `self` + the kwargs the route depends on
        assert "image_path" in param_names
        assert "height_cm" in param_names
        assert "reference_object_cm" in param_names
        assert "gender_profile" in param_names
        assert "subject_id" in param_names

    def test_pipeline_run_with_mesh_returns_pipeline_result(self) -> None:
        """The MeasurementService method delegates to
        `PipelineOrchestrator.run_with_mesh`. That method must return a
        PipelineResult (not a bare BodyMeasurements) so callers can
        access `.mesh`, `.betas`, `.thetas`."""
        from typing import get_type_hints

        assert hasattr(PipelineOrchestrator, "run_with_mesh")
        # `from __future__ import annotations` lazifies annotations to
        # strings, so resolve them via get_type_hints.
        hints = get_type_hints(PipelineOrchestrator.run_with_mesh)
        assert hints.get("return") is PipelineResult


class TestH13MeshRescaling:
    """When the user supplies a height, the SMPL mesh must be rescaled
    from template space (~1.7m) to that height before MeshAnalyzer runs.
    Without this every output clusters around SMPL defaults regardless
    of user input — exactly what audit FS-H13 documented."""

    def test_rescale_returns_mesh_unchanged_when_no_target_height(self) -> None:
        original_verts = np.array([[0.0, 0.0, 0.0], [0.0, 1.7, 0.0]])
        mesh = SimpleNamespace(vertices=original_verts, faces=np.array([[0, 1, 0]]))
        scale = ScaleFactor(
            pixels_per_cm=5.0,
            source="reference_object",
            confidence=0.9,
            target_height_cm=None,
        )

        out = PipelineOrchestrator._rescale_mesh_to_height(mesh, scale)
        # Same object back, vertices untouched
        assert out is mesh

    def test_rescale_scales_mesh_to_target_height(self) -> None:
        # Template-scale mesh: 1.7m tall (apex at y=1.7)
        original_verts = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 1.7, 0.0],
                [-0.5, 1.7, 0.0],
            ]
        )
        mesh = SimpleNamespace(
            vertices=original_verts.copy(),
            faces=np.array([[0, 1, 2]]),
        )
        scale = ScaleFactor(
            pixels_per_cm=5.0,
            source="user_height",
            confidence=0.95,
            target_height_cm=180.0,  # User is 1.80m
        )

        out = PipelineOrchestrator._rescale_mesh_to_height(mesh, scale)

        # Output mesh's vertical span should be the user's height in metres
        verts = np.asarray(out.vertices)
        new_height_m = verts[:, 1].max() - verts[:, 1].min()
        assert new_height_m == pytest.approx(1.80, rel=1e-4)

    def test_rescale_preserves_horizontal_proportions(self) -> None:
        """Scaling is uniform — width-to-height ratio of the mesh is
        invariant. Otherwise the bust/waist circumferences come out
        anatomically wrong even if the height is right."""
        original_verts = np.array(
            [
                [-0.4, 0.0, 0.0],  # left foot
                [0.4, 0.0, 0.0],  # right foot
                [0.0, 1.7, 0.0],  # head
            ]
        )
        mesh = SimpleNamespace(
            vertices=original_verts.copy(),
            faces=np.array([[0, 1, 2]]),
        )
        scale = ScaleFactor(
            pixels_per_cm=5.0,
            source="user_height",
            confidence=0.95,
            target_height_cm=200.0,
        )

        out = PipelineOrchestrator._rescale_mesh_to_height(mesh, scale)
        verts = np.asarray(out.vertices)

        # Original aspect ratio: width 0.8 / height 1.7 = 0.4706
        # Scaled aspect ratio must match
        new_width = verts[:, 0].max() - verts[:, 0].min()
        new_height = verts[:, 1].max() - verts[:, 1].min()
        assert new_width / new_height == pytest.approx(0.8 / 1.7, rel=1e-4)
