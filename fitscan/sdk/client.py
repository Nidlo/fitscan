"""
FitScan SDK — Python client for consuming apps.
Usage:
    from fitscan.sdk.client import FitScanClient
    client = FitScanClient(base_url="http://fitscan-api:8000")
    result = client.measure("photo.jpg", height_cm=165.0)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import httpx


class FitScanClient:
    """HTTP client for the FitScan API. No business logic — HTTP calls only."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 60.0,
    ) -> None:
        self.base = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    def measure(
        self,
        image_path: str | Path,
        height_cm: Optional[float] = None,
        reference_object_cm: Optional[float] = None,
        gender: str = "ladies",
        enable_smpl: bool = False,
        enable_claude: bool = True,
    ) -> dict:
        """Submit a photo for body measurement."""
        with open(image_path, "rb") as f:
            files = {"image": (Path(image_path).name, f, "image/jpeg")}
            data: dict[str, str] = {
                "gender_profile": gender,
                "enable_smpl": str(enable_smpl).lower(),
                "enable_claude": str(enable_claude).lower(),
            }
            if height_cm:
                data["height_cm"] = str(height_cm)
            if reference_object_cm:
                data["reference_object_cm"] = str(reference_object_cm)
            r = self._client.post(f"{self.base}/api/v1/measure", files=files, data=data)
        r.raise_for_status()
        return r.json()

    def get_avatar_glb(self, avatar_id: str) -> bytes:
        """Download avatar GLB binary."""
        r = self._client.get(f"{self.base}/api/v1/avatar/{avatar_id}/glb")
        r.raise_for_status()
        return r.content

    def get_avatar_thumbnail(self, avatar_id: str) -> bytes:
        """Download avatar 512x512 thumbnail PNG."""
        r = self._client.get(f"{self.base}/api/v1/avatar/{avatar_id}/thumbnail")
        r.raise_for_status()
        return r.content

    def fit_garment(self, avatar_id: str, garment_type: str) -> bytes:
        """Drape a garment on the avatar. Returns combined GLB bytes."""
        r = self._client.post(
            f"{self.base}/api/v1/avatar/{avatar_id}/fit-garment",
            data={"garment_type": garment_type},
        )
        r.raise_for_status()
        return r.content

    def override_measurements(
        self, avatar_id: str, overrides: dict[str, float]
    ) -> dict:
        """Submit manual measurement corrections."""
        r = self._client.post(
            f"{self.base}/api/v1/avatar/{avatar_id}/override",
            json={"overrides": overrides},
        )
        r.raise_for_status()
        return r.json()

    def health(self) -> bool:
        try:
            r = self._client.get(f"{self.base}/api/v1/health")
            return r.status_code == 200
        except Exception:
            return False

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> FitScanClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
