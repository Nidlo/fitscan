"""Tests for configuration module."""

from fitscan.config import Settings, get_settings


class TestSettings:
    def test_default_values(self):
        s = Settings(app_env="test", database_url="sqlite:///test.db")
        assert s.app_env == "test"
        assert s.confidence_threshold == 0.65
        assert s.max_image_size_mb == 10

    def test_is_test(self):
        s = Settings(app_env="test", database_url="sqlite:///test.db")
        assert s.is_test
        assert not s.is_production

    def test_is_production(self):
        s = Settings(app_env="production", database_url="sqlite:///test.db")
        assert s.is_production
        assert not s.is_test

    def test_get_settings_returns_settings(self):
        get_settings.cache_clear()
        s = get_settings()
        assert isinstance(s, Settings)
