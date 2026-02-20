"""
yolo_detector.py – YOLOv8 pothole detection on 2-D depth images.

ARM / Raspberry Pi safety design
──────────────────────────────────
PyTorch wheels compiled for x86 AVX2 or ARMv8.2+ (unsupported on Pi 4B's
Cortex-A72 / ARMv8.0) raise SIGILL — a hardware-level signal that Python's
try/except CANNOT catch.  The entire process is killed.

To avoid this we run a one-shot subprocess test before ever importing torch
in the main process.  If the subprocess exits with any non-zero code
(including -4 / 132 which is SIGILL on Linux), YOLO is disabled for the
session and the system runs in heuristic-only mode at full speed.

Heuristic-only mode
────────────────────
When YOLO is unavailable, detect() and detect_from_profile() return [].
The LiDAR depth-threshold event tracker (PotholeEventTracker) handles all
detection.  Results are reported normally to the backend.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger("PotholeSystem.YOLODetector")

# ── One-shot subprocess safety probe ─────────────────────────────────────────
_YOLO_CLASS      = None   # ultralytics.YOLO class (when available)
_YOLO_AVAILABLE: Optional[bool] = None   # None = not yet probed


def _probe_yolo_safe() -> bool:
    """
    Import-test ultralytics in a throwaway subprocess.

    If torch is compiled for an unsupported CPU, the subprocess dies with
    SIGILL (returncode -4 / 132) and this function returns False — leaving
    the main process completely unharmed.

    Returns True only when the subprocess exits cleanly and prints 'yolo_ok'.
    """
    global _YOLO_CLASS, _YOLO_AVAILABLE

    if _YOLO_AVAILABLE is not None:
        return _YOLO_AVAILABLE

    logger.info("Probing YOLO availability (subprocess safety test) …")
    _test_script = (
        "from ultralytics import YOLO; "
        "m = YOLO.__new__(YOLO); "   # don't load weights — just test import
        "print('yolo_ok')"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", _test_script],
            capture_output=True,
            timeout=30,
            text=True,
        )
        if result.returncode == 0 and "yolo_ok" in result.stdout:
            # Safe to import in the main process
            try:
                from ultralytics import YOLO  # noqa: PLC0415
                _YOLO_CLASS = YOLO
                _YOLO_AVAILABLE = True
                logger.info("✓ ultralytics / torch probed OK — YOLO enabled")
            except Exception as exc:
                _YOLO_AVAILABLE = False
                logger.warning(f"YOLO import OK in subprocess but failed here: {exc}")
        else:
            _YOLO_AVAILABLE = False
            sig = result.returncode
            stderr_snippet = result.stderr.strip()[:300] if result.stderr else "—"

            if sig in (-4, 132):
                hint = (
                    "PyTorch is compiled for an unsupported CPU instruction set "
                    "(SIGILL on ARM Cortex-A72 / Raspberry Pi 4B).\n"
                    "  Fix:\n"
                    "    sudo pip3 uninstall torch torchvision torchaudio -y\n"
                    "    pip install onnxruntime   # lightweight ARM inference\n"
                    "    pip install torch==2.1.2 --index-url "
                    "https://download.pytorch.org/whl/cpu"
                )
            else:
                hint = (
                    "Run:  pip install torch==2.1.2 "
                    "--index-url https://download.pytorch.org/whl/cpu "
                    "&& pip install ultralytics"
                )

            logger.warning(
                f"YOLO disabled (subprocess exit={sig}).\n"
                f"  stderr: {stderr_snippet}\n"
                f"  {hint}\n"
                "  Continuing in heuristic-only mode."
            )

    except subprocess.TimeoutExpired:
        _YOLO_AVAILABLE = False
        logger.warning("YOLO probe timed out — YOLO disabled.")
    except Exception as exc:
        _YOLO_AVAILABLE = False
        logger.warning(f"YOLO probe error ({exc}) — YOLO disabled.")

    return _YOLO_AVAILABLE


# ═══════════════════════════════════════════════════════════════════════════════
#  Data types
# ═══════════════════════════════════════════════════════════════════════════════

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
        self.bbox       = bbox    # (x1, y1, x2, y2) in image-pixel coords
        self.class_id   = class_id
        self.label      = label

    def __repr__(self) -> str:
        return (
            f"DetectionResult(label={self.label!r}, "
            f"conf={self.confidence:.2f}, bbox={self.bbox})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Detector
# ═══════════════════════════════════════════════════════════════════════════════

class YOLOPotholeDetector:
    """
    Wraps YOLOv8 for real-time pothole detection on LiDAR depth images.

    Fully operational even without YOLO: returns [] for every inference call
    and logs the reason once.  Check `detector.yolo_available` for status.
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
        self._model_path      = Path(model_path)
        self._pretrained_base = pretrained_base
        self._conf            = confidence
        self._iou             = iou
        self._device          = device
        self._image_size      = image_size

        self._model             = None
        self._model_version     = 0
        self._total_inferences  = 0
        self._total_detections  = 0

        self._load_model()

    # ── model lifecycle ───────────────────────────────────────────────────────

    def _load_model(self) -> None:
        if not _probe_yolo_safe():
            return   # stay in heuristic-only mode

        YOLO = _YOLO_CLASS
        # Try custom checkpoint first
        if self._model_path.exists():
            try:
                self._model = YOLO(str(self._model_path))
                logger.info(f"✓ YOLO model loaded from {self._model_path}")
                return
            except Exception as exc:
                logger.warning(f"Custom model load failed ({exc}); using base")

        # Fallback: download/use pretrained base
        try:
            logger.info(f"  Loading pretrained base: {self._pretrained_base} …")
            self._model = YOLO(self._pretrained_base)
            self._model_path.parent.mkdir(parents=True, exist_ok=True)
            self._model.save(str(self._model_path))
            logger.info(f"✓ Base model saved → {self._model_path}")
        except Exception as exc:
            logger.error(f"Base model load failed ({exc}). YOLO disabled.")
            self._model = None

    def reload_model(self) -> None:
        """Hot-reload checkpoint (called by OnlineTrainer after retraining)."""
        if not _YOLO_AVAILABLE or _YOLO_CLASS is None:
            return
        try:
            self._model     = _YOLO_CLASS(str(self._model_path))
            self._model_version += 1
            logger.info(f"✓ YOLO model reloaded (v{self._model_version})")
        except Exception as exc:
            logger.error(f"Model reload failed: {exc}")

    # ── inference ─────────────────────────────────────────────────────────────

    @property
    def yolo_available(self) -> bool:
        return self._model is not None

    def detect(self, image_rgb: np.ndarray) -> List[DetectionResult]:
        """
        Run YOLOv8 on a [H, W, 3] uint8 RGB image.
        Returns [] if YOLO is unavailable or image is None.
        """
        if self._model is None or image_rgb is None:
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

        logger.debug(f"Inference: {(time.perf_counter()-t0)*1000:.1f} ms")

        detections: List[DetectionResult] = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            for i in range(len(boxes)):
                conf  = float(boxes.conf[i])
                cls   = int(boxes.cls[i])
                xyxy  = boxes.xyxy[i].tolist()
                label = (result.names.get(cls, "pothole")
                         if result.names else "pothole")
                detections.append(
                    DetectionResult(confidence=conf, bbox=tuple(xyxy),
                                    class_id=cls, label=label)
                )

        self._total_detections += len(detections)
        if detections:
            best = max(detections, key=lambda d: d.confidence)
            logger.info(
                f"🎯 YOLO: {len(detections)} detection(s) "
                f"[best conf={best.confidence:.2f}]"
            )
        return detections

    def detect_from_profile(
        self,
        depth_profile: List[float],
        image_width: int = 64,
        image_height: int = 64,
    ) -> List[DetectionResult]:
        """Build a depth image from a 1-D profile and run detect()."""
        if not depth_profile or not self.yolo_available:
            return []

        arr       = np.array(depth_profile, dtype=np.float32)
        x_src     = np.linspace(0, 1, len(arr))
        x_dst     = np.linspace(0, 1, image_width)
        resampled = np.interp(x_dst, x_src, arr).astype(np.float32)
        max_val   = resampled.max()
        norm      = (
            ((resampled / max_val) * 255).astype(np.uint8)
            if max_val > 0.01 else np.zeros(image_width, dtype=np.uint8)
        )
        image_2d  = np.tile(norm, (image_height, 1))
        image_rgb = np.stack([image_2d, image_2d, image_2d], axis=-1)
        return self.detect(image_rgb)

    # ── stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        return {
            "model_version":    self._model_version,
            "total_inferences": self._total_inferences,
            "total_detections": self._total_detections,
            "yolo_available":   self.yolo_available,
            "detection_rate": (
                self._total_detections / max(1, self._total_inferences)
            ),
        }

    @property
    def model_version(self) -> int:
        return self._model_version
