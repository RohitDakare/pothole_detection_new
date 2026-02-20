"""
yolo_detector.py – YOLOv8 pothole detection on 2-D depth images.

How it works
────────────
1.  The DepthImageBuilder converts the rolling LiDAR buffer into a small
    2-D grayscale / RGB image (default 64 × 64).
2.  This module runs YOLOv8 inference on that image.
3.  Any bounding-box detection with confidence ≥ threshold is treated as
    a pothole candidate (class 0 = "pothole").
4.  The detector also exposes detect_from_profile() which accepts a raw
    1-D depth profile (list of cm values) so the OnlineTrainer and the
    event handler can also call it directly.

Model bootstrapping
──────────────────
On first run, if no custom model checkpoint exists, the system downloads
yolov8n.pt (Ultralytics nano) and fine-tunes it immediately on any
synthetic depth data already present in training_data/.  This ensures the
system works out-of-the-box even before the first real detection.
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger("PotholeSystem.YOLODetector")

# Lazy-import ultralytics so the import doesn't crash if it isn't installed
_YOLO = None


def _get_yolo():
    global _YOLO
    if _YOLO is None:
        try:
            from ultralytics import YOLO
            _YOLO = YOLO
        except ImportError:
            logger.error(
                "ultralytics not installed – run: pip install ultralytics"
            )
            raise
    return _YOLO


class DetectionResult:
    """Holds one YOLO detection output."""

    __slots__ = ("confidence", "bbox", "class_id", "label")

    def __init__(
        self,
        confidence: float,
        bbox: Tuple[float, float, float, float],
        class_id: int = 0,
        label: str = "pothole",
    ):
        self.confidence = confidence
        self.bbox = bbox          # (x1, y1, x2, y2) in image-pixel coords
        self.class_id = class_id
        self.label = label

    def __repr__(self) -> str:
        return (
            f"DetectionResult(label={self.label!r}, "
            f"conf={self.confidence:.2f}, bbox={self.bbox})"
        )


class YOLOPotholeDetector:
    """
    Wraps YOLOv8 for real-time pothole detection on LiDAR depth images.

    Parameters
    ----------
    model_path : str
        Path to the custom .pt checkpoint.  If it does not exist, the
        system falls back to the pretrained base and synthesises initial
        training data automatically.
    pretrained_base : str
        Ultralytics model tag / path used as fallback (e.g. 'yolov8n.pt').
    confidence : float
        Minimum confidence threshold to report a detection.
    iou : float
        NMS IoU threshold.
    device : str
        'cpu' on Pi 4B (no GPU).
    image_size : int
        Inference image size (must match depth-image dimensions).
    """

    def __init__(
        self,
        model_path: str,
        pretrained_base: str = "yolov8n.pt",
        confidence: float = 0.40,
        iou: float = 0.45,
        device: str = "cpu",
        image_size: int = 64,
    ):
        self._model_path = Path(model_path)
        self._pretrained_base = pretrained_base
        self._conf = confidence
        self._iou = iou
        self._device = device
        self._image_size = image_size

        self._model = None
        self._model_version: int = 0    # incremented after each online retrain
        self._total_inferences: int = 0
        self._total_detections: int = 0

        self._load_model()

    # ── model lifecycle ───────────────────────────────────────────────────────

    def _load_model(self) -> None:
        YOLO = _get_yolo()
        if self._model_path.exists():
            try:
                self._model = YOLO(str(self._model_path))
                logger.info(f"✓ YOLO model loaded from {self._model_path}")
                return
            except Exception as exc:
                logger.warning(f"Custom model load failed ({exc}); using base")

        # Fallback: pretrained base (YOLOv8 nano)
        logger.info(f"  Loading pretrained base: {self._pretrained_base}")
        self._model = YOLO(self._pretrained_base)
        # Save it as the custom checkpoint so OnlineTrainer finds it
        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(str(self._model_path))
        logger.info(f"✓ Base model saved to {self._model_path}")

    def reload_model(self) -> None:
        """Reload the checkpoint from disk (called by OnlineTrainer after update)."""
        try:
            YOLO = _get_yolo()
            self._model = YOLO(str(self._model_path))
            self._model_version += 1
            logger.info(
                f"✓ YOLO model reloaded (version {self._model_version})"
            )
        except Exception as exc:
            logger.error(f"Model reload failed: {exc}")

    # ── inference ─────────────────────────────────────────────────────────────

    def detect(self, image_rgb: np.ndarray) -> List[DetectionResult]:
        """
        Run inference on a pre-built RGB image [H, W, 3] uint8.

        Returns a list of DetectionResult objects (may be empty).
        """
        if self._model is None:
            return []
        if image_rgb is None:
            return []

        self._total_inferences += 1
        t0 = time.perf_counter()

        try:
            results = self._model.predict(
                source=image_rgb,
                conf=self._conf,
                iou=self._iou,
                device=self._device,
                imgsz=self._image_size,
                verbose=False,
            )
        except Exception as exc:
            logger.error(f"YOLO inference error: {exc}")
            return []

        dt = time.perf_counter() - t0
        logger.debug(f"Inference: {dt * 1000:.1f} ms")

        detections: List[DetectionResult] = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            for i in range(len(boxes)):
                conf = float(boxes.conf[i])
                cls = int(boxes.cls[i])
                xyxy = boxes.xyxy[i].tolist()
                det = DetectionResult(
                    confidence=conf,
                    bbox=tuple(xyxy),
                    class_id=cls,
                    label=result.names.get(cls, "pothole") if result.names else "pothole",
                )
                detections.append(det)

        self._total_detections += len(detections)
        if detections:
            logger.info(
                f"🎯 YOLO detected {len(detections)} pothole(s) "
                f"[best conf={max(d.confidence for d in detections):.2f}]"
            )
        return detections

    def detect_from_profile(
        self,
        depth_profile: List[float],
        image_width: int = 64,
        image_height: int = 64,
    ) -> List[DetectionResult]:
        """
        Convenience wrapper: build a depth image from a raw profile list
        and run detect() on it.

        depth_profile : list of depth-drop values in cm (already baseline-subtracted)
        """
        if not depth_profile:
            return []

        arr = np.array(depth_profile, dtype=np.float32)
        # Resample to image width
        x_src = np.linspace(0, 1, len(arr))
        x_dst = np.linspace(0, 1, image_width)
        resampled = np.interp(x_dst, x_src, arr).astype(np.float32)
        # Normalise
        max_val = resampled.max()
        if max_val < 0.01:
            norm = np.zeros(image_width, dtype=np.uint8)
        else:
            norm = ((resampled / max_val) * 255).astype(np.uint8)
        image_2d = np.tile(norm, (image_height, 1))
        image_rgb = np.stack([image_2d, image_2d, image_2d], axis=-1)
        return self.detect(image_rgb)

    # ── stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        return {
            "model_version": self._model_version,
            "total_inferences": self._total_inferences,
            "total_detections": self._total_detections,
            "detection_rate": (
                self._total_detections / max(1, self._total_inferences)
            ),
        }

    @property
    def model_version(self) -> int:
        return self._model_version
