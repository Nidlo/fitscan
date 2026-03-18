"""
Stores and retrieves binary files (GLB, thumbnails, source images).
Backends: local filesystem (test/dev), MinIO (local prod), AWS S3 (cloud prod).
"""

from __future__ import annotations

import io
from pathlib import Path

from fitscan.config import get_settings


class ObjectStore:
    """Unified interface over local fs / MinIO / S3. Single responsibility."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._backend = self._settings.storage_backend

    def put(
        self,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Store bytes at key. Returns the public URL."""
        if self._backend == "local":
            return self._put_local(key, data)
        elif self._backend == "minio":
            return self._put_minio(key, data, content_type)
        else:
            return self._put_s3(key, data, content_type)

    def get(self, key: str) -> bytes:
        if self._backend == "local":
            return self._get_local(key)
        elif self._backend == "minio":
            return self._get_minio(key)
        else:
            return self._get_s3(key)

    def _put_local(self, key: str, data: bytes) -> str:
        path = Path("data/avatars") / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return f"/local/{key}"

    def _get_local(self, key: str) -> bytes:
        return (Path("data/avatars") / key).read_bytes()

    def _put_minio(self, key: str, data: bytes, content_type: str) -> str:
        from minio import Minio

        s = self._settings
        client = Minio(
            s.minio_endpoint, s.minio_access_key, s.minio_secret_key, secure=False
        )
        if not client.bucket_exists(s.minio_bucket):
            client.make_bucket(s.minio_bucket)
        client.put_object(
            s.minio_bucket,
            key,
            io.BytesIO(data),
            len(data),
            content_type=content_type,
        )
        return f"http://{s.minio_endpoint}/{s.minio_bucket}/{key}"

    def _get_minio(self, key: str) -> bytes:
        from minio import Minio

        s = self._settings
        client = Minio(
            s.minio_endpoint, s.minio_access_key, s.minio_secret_key, secure=False
        )
        response = client.get_object(s.minio_bucket, key)
        return response.read()

    def _put_s3(self, key: str, data: bytes, content_type: str) -> str:
        import boto3

        s = self._settings
        client = boto3.client("s3", region_name=s.s3_region)
        client.put_object(
            Bucket=s.s3_bucket, Key=key, Body=data, ContentType=content_type
        )
        return f"https://{s.s3_bucket}.s3.{s.s3_region}.amazonaws.com/{key}"

    def _get_s3(self, key: str) -> bytes:
        import boto3

        s = self._settings
        client = boto3.client("s3")
        return client.get_object(Bucket=s.s3_bucket, Key=key)["Body"].read()
