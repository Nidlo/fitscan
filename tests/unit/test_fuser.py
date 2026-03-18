"""Tests for measurement fusion."""
import pytest

from fitscan.domain.models import BodyMeasurements, SingleMeasurement
from fitscan.fusion.fuser import MeasurementFuser


def make_measurement(
    value: float, confidence: float, source: str
) -> SingleMeasurement:
    return SingleMeasurement(
        value_cm=value, confidence=confidence, source=source
    )


class TestMeasurementFuser:
    fuser = MeasurementFuser()

    def test_fuse_single_source_returns_same(self):
        m = BodyMeasurements(
            chest_cm=make_measurement(95.0, 0.8, "smpl"),
            source="smpl",
        )
        result = self.fuser.fuse([m])
        assert result.chest_cm is not None
        assert result.chest_cm.value_cm == 95.0

    def test_fuse_two_sources_weighted_average(self):
        m_smpl = BodyMeasurements(
            chest_cm=make_measurement(96.0, 0.9, "smpl"),
            source="smpl",
        )
        m_sil = BodyMeasurements(
            chest_cm=make_measurement(90.0, 0.7, "silhouette"),
            source="silhouette",
        )
        result = self.fuser.fuse([m_smpl, m_sil])
        assert result.chest_cm is not None
        assert 93.0 <= result.chest_cm.value_cm <= 97.0
        assert result.source == "fused"

    def test_fuse_none_field_returns_available(self):
        m1 = BodyMeasurements(
            chest_cm=make_measurement(95.0, 0.8, "smpl"),
            source="smpl",
        )
        m2 = BodyMeasurements(chest_cm=None, source="silhouette")
        result = self.fuser.fuse([m1, m2])
        assert result.chest_cm is not None

    def test_fuse_empty_raises(self):
        with pytest.raises(ValueError):
            self.fuser.fuse([])

    def test_fuse_preserves_gender_profile(self):
        m = BodyMeasurements(
            gender_profile="men",
            chest_cm=make_measurement(100.0, 0.9, "smpl"),
            source="smpl",
        )
        result = self.fuser.fuse([m])
        assert result.gender_profile == "men"
