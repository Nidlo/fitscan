"""Background removal and image preprocessing. Implements IImagePreprocessor."""

from __future__ import annotations

import cv2
import numpy as np
from loguru import logger
from PIL import Image

from fitscan.domain.models import ProcessedImage


class ImagePreprocessor:
    """Removes background, extracts silhouette, normalises image."""

    def remove_background(self, image: np.ndarray) -> np.ndarray:
        """Returns RGBA image with background removed."""
        try:
            from rembg import remove as rembg_remove

            pil_img = Image.fromarray(image)
            result = rembg_remove(pil_img)
            return np.array(result)
        except (ImportError, SystemExit, Exception) as e:
            logger.warning("rembg not available ({}), using threshold-based removal.", e)
            return self._threshold_remove(image)

    def extract_silhouette(self, image: np.ndarray) -> np.ndarray:
        """Returns binary mask: 255=person, 0=background."""
        if len(image.shape) > 2 and image.shape[2] == 4:
            alpha = image[:, :, 3]
            _, mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def process(self, image: np.ndarray) -> ProcessedImage:
        no_bg = self.remove_background(image)
        mask = self.extract_silhouette(no_bg)
        rgb = no_bg[:, :, :3] if no_bg.shape[2] == 4 else no_bg
        h, w = image.shape[:2]
        return ProcessedImage(
            original=image,
            no_background=rgb,
            silhouette_mask=mask,
            width=w,
            height=h,
        )

    def _threshold_remove(self, image: np.ndarray) -> np.ndarray:
        """Fallback background removal using simple thresholding."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        rgba = np.dstack([image, mask])
        return rgba
