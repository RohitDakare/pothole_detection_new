"""
depth_image.py – Rolling LiDAR buffer → 2D depth-image conversion.

The TF-02 Pro outputs a 1-D stream of distance values at up to 100 Hz.
To feed data into YOLOv8 (which expects 2-D spatial input), we:

  1. Maintain a fixed-length circular buffer of raw readings.
  2. Track a dynamic road-surface baseline using a rolling low-percentile.
  3. Convert depth-drop values to a normalised 2-D grayscale "depth image"
     where:
        x-axis  → time (latest sample on the right)
        y-axis  → depth bin (deeper dips shown brighter)
  4. Tile the 1-D depth signal into H rows so YOLOv8 receives an HxW tensor.
     A pothole appears as a bright rectangle — exactly the bounding-box 
     pattern YOLO was designed to detect.

The image representation also makes it trivial to visualise and debug 
detection quality without a camera.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("PotholeSystem.DepthImage")


class DepthImageBuilder:
    """
    Thread-safe rolling buffer that converts LiDAR readings into
    normalised 2-D depth images suitable for YOLOv8 inference.

    Parameters
    ----------
    buffer_size : int
        Number of LiDAR samples to keep in the circular window.
        At 100 Hz this equals 2 seconds of history.
    image_width : int
        Width  (time axis) of the output image in pixels.
    image_height : int
        Height (depth axis) of the output image in pixels.
    sensor_height_cm : float
        Physical height of the LiDAR above the road surface.
        Raw readings ≥ sensor_height_cm → depth ≈ 0 (flat road).
    baseline_percentile : float
        Low-percentile of the rolling window used to estimate the
        reference road-surface distance. 10-th → 20-th percentile
        works well for typical road surfaces.
    """

    def __init__(
        self,
        buffer_size: int = 200,
        image_width: int = 64,
        image_height: int = 64,
        sensor_height_cm: float = 20.0,
        baseline_percentile: float = 15.0,
    ):
        self._buf: deque[float] = deque(maxlen=buffer_size)
        self._buffer_size = buffer_size
        self._img_w = image_width
        self._img_h = image_height
        self._sensor_height_cm = sensor_height_cm
        self._baseline_pct = baseline_percentile

        self._lock = threading.Lock()
        self._baseline: float = sensor_height_cm   # initialise to sensor height

    # ── public API ─────────────────────────────────────────────────────────

    def push(self, raw_cm: float) -> None:
        """
        Add a new raw LiDAR reading (centimetres from sensor head).
        Thread-safe; O(1).
        """
        with self._lock:
            self._buf.append(raw_cm)
            # Update rolling baseline incrementally
            if len(self._buf) >= 10:
                arr = np.array(self._buf)
                self._baseline = float(np.percentile(arr, self._baseline_pct))

    def get_depth_cm(self, raw_cm: float) -> float:
        """
        Convert a raw reading to a pothole depth in cm.
        Depth = max(0, raw_cm - baseline).
        A positive depth means the LiDAR sees *further* than baseline
        (i.e., looking into a hole).
        """
        with self._lock:
            baseline = self._baseline
        depth = raw_cm - baseline
        return max(0.0, depth)

    def get_baseline_cm(self) -> float:
        """Return the current estimated road-surface baseline distance."""
        with self._lock:
            return self._baseline

    def build_image(self) -> Optional[np.ndarray]:
        """
        Build a normalised uint8 grayscale image [H, W] from the rolling buffer.

        Returns None if not enough samples collected yet.
        The image is scaled so that:
          • 0 px → flat road (depth = 0)
          • 255 px → maximum observed depth in this window
        """
        with self._lock:
            if len(self._buf) < self._img_w:
                return None
            arr = np.array(self._buf, dtype=np.float32)
            baseline = self._baseline

        # Compute depth drop (positive = hole, negative = bump → clip to 0)
        depth = np.clip(arr - baseline, 0.0, None)

        # Resample to image width via linear interpolation
        x_original = np.linspace(0, 1, len(depth))
        x_target = np.linspace(0, 1, self._img_w)
        resampled = np.interp(x_target, x_original, depth).astype(np.float32)

        # Normalise 0-255
        max_depth = resampled.max()
        if max_depth < 0.1:          # All-flat road (no pothole)
            norm = np.zeros(self._img_w, dtype=np.uint8)
        else:
            norm = ((resampled / max_depth) * 255).astype(np.uint8)

        # Tile 1-D signal into 2-D image (H rows = same 1-D signal repeated)
        # This gives YOLO a "stripe" pattern it can bound-box
        image = np.tile(norm, (self._img_h, 1))   # shape: (H, W)
        return image

    def build_rgb_image(self) -> Optional[np.ndarray]:
        """
        Build a 3-channel RGB image [H, W, 3] (required by YOLOv8).
        All three channels carry the same grayscale signal.
        Returns None if insufficient data.
        """
        gray = self.build_image()
        if gray is None:
            return None
        return np.stack([gray, gray, gray], axis=-1)   # (H, W, 3)

    def get_rolling_stats(self) -> dict:
        """Return summary statistics of the rolling buffer."""
        with self._lock:
            if len(self._buf) == 0:
                return {}
            arr = np.array(self._buf)
        return {
            "count": len(arr),
            "mean_cm": float(arr.mean()),
            "min_cm": float(arr.min()),
            "max_cm": float(arr.max()),
            "baseline_cm": float(self._baseline),
            "max_depth_cm": float(max(0, arr.max() - self._baseline)),
        }

    def get_window_copy(self) -> np.ndarray:
        """Return a numpy copy of the current rolling window."""
        with self._lock:
            return np.array(self._buf, dtype=np.float32)


class PotholeEventTracker:
    """
    State machine that turns a stream of per-sample depth values into
    discrete pothole-event windows.

    Transitions:
        IDLE ──(depth > threshold)-→ IN_EVENT
        IN_EVENT ──(depth ≤ threshold OR timeout)-→ IDLE  [fires callback]
        IN_EVENT ──(depth > threshold, keep accumulating)

    Usage::

        def on_pothole(readings, start_t, gps):
            ...

        tracker = PotholeEventTracker(threshold_cm=3.0, on_event=on_pothole)
        # call tracker.feed(depth_cm, timestamp, gps_location) in the hot loop
    """

    def __init__(
        self,
        threshold_cm: float = 3.0,
        min_samples: int = 5,
        max_duration_s: float = 3.0,
        on_event=None,
    ):
        self._threshold = threshold_cm
        self._min_samples = min_samples
        self._max_duration = max_duration_s
        self._on_event = on_event or (lambda readings, start_t, gps: None)

        self._in_event = False
        self._readings: list = []
        self._start_time: float = 0.0
        self._last_gps: dict = {}

    def feed(self, depth_cm: float, timestamp: float, gps: dict) -> bool:
        """
        Feed one depth sample.

        Returns True when an event just fired (pothole confirmed).
        """
        self._last_gps = gps

        if depth_cm > self._threshold:
            if not self._in_event:
                # ── Event START ───────────────────────────────────────────
                self._in_event = True
                self._start_time = timestamp
                self._readings = []
            self._readings.append(depth_cm)

            # Timeout guard: sensor lifted or very long feature → reset
            if (timestamp - self._start_time) > self._max_duration:
                logger.warning(
                    f"Event timeout ({self._max_duration:.1f} s) — resetting"
                )
                self._in_event = False
                self._readings = []
            return False

        elif self._in_event:
            # ── Event END ─────────────────────────────────────────────────
            self._in_event = False
            readings_copy = list(self._readings)
            self._readings = []

            if len(readings_copy) >= self._min_samples:
                self._on_event(readings_copy, self._start_time, self._last_gps)
                return True
            else:
                logger.debug(
                    f"Discarded short event ({len(readings_copy)} samples "
                    f"< min {self._min_samples})"
                )
        return False

    @property
    def in_event(self) -> bool:
        return self._in_event
