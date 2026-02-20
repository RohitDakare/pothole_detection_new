"""
main.py – Optimized Pothole Detection System (Entry Point)
Hardware: Raspberry Pi 4B + Benewake TF-02 Pro LiDAR + u-blox NEO-6M GPS

Architecture
────────────
Thread 1  [main]        : 100 Hz LiDAR polling + depth event tracking
Thread 2  [GPS-Reader]  : background NMEA parse (inside NEO6MGPS)
Thread 3  [LiDAR-Reader]: background frame parse (inside TF02ProLiDAR)
Thread 4  [YOLO-Trainer]: periodic model fine-tune (inside OnlineTrainer)
Thread 5  [BackendWorker]: HTTP event dispatch (inside BackendClient)
Thread 6  [ProfileUpload]: road-profile telemetry (inside BackendClient)

Signal handling: SIGINT / SIGTERM → graceful shutdown of all threads.
"""

from __future__ import annotations

import logging
import logging.handlers
import math
import os
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ── Allow importing project modules regardless of CWD ─────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from config import SystemConfig
from sensors import TF02ProLiDAR, NEO6MGPS
from depth_image import DepthImageBuilder, PotholeEventTracker
from yolo_detector import YOLOPotholeDetector
from online_trainer import OnlineTrainer
from backend_client import BackendClient


# ═══════════════════════════════════════════════════════════════════════════════
#  Logging setup
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging(cfg: SystemConfig) -> logging.Logger:
    log_dir = Path(cfg.logging.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, cfg.logging.log_level, logging.INFO))

    fmt_detail = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)-30s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fmt_simple = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Rotating file – everything
    fh = logging.handlers.RotatingFileHandler(
        log_dir / "pothole_system.log",
        maxBytes=cfg.logging.max_bytes,
        backupCount=cfg.logging.backup_count,
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt_detail)

    # Rotating file – detections only
    dh = logging.handlers.RotatingFileHandler(
        log_dir / "detections.log",
        maxBytes=cfg.logging.max_bytes,
        backupCount=cfg.logging.backup_count,
    )
    dh.setLevel(logging.INFO)
    dh.setFormatter(fmt_detail)
    dh.addFilter(lambda r: "DETECTION" in r.getMessage() or "POTHOLE" in r.getMessage().upper())

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt_simple)

    root.addHandler(fh)
    root.addHandler(dh)
    root.addHandler(ch)

    return logging.getLogger("PotholeSystem")


# ═══════════════════════════════════════════════════════════════════════════════
#  Measurement helpers
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_pothole_dimensions(
    readings_cm: List[float],
    duration_s: float,
    vehicle_speed_cms: float,
    sensor_height_cm: float,
) -> Dict:
    """
    Estimate real-world pothole dimensions from LiDAR depth readings.

    Returns dict with:  depth, length, width, volume, severity, avg_depth
    """
    if not readings_cm:
        return {}

    raw = [max(0.0, r) for r in readings_cm]
    max_depth = max(raw)
    avg_depth = sum(raw) / len(raw)

    # Length = time vehicle spent over the hole × speed
    length = duration_s * vehicle_speed_cms

    # Width heuristic: typically 70–90 % of length for road potholes
    width = length * 0.80

    # Volume: approximate as half an ellipsoid
    volume = (math.pi / 6.0) * length * width * max_depth

    # Severity
    if max_depth < 3.0:
        severity = "Minor"
        status = "Orange"
    elif max_depth < 8.0:
        severity = "Moderate"
        status = "Orange"
    else:
        severity = "Critical"
        status = "Red"

    return {
        "depth": round(max_depth, 2),
        "avg_depth": round(avg_depth, 2),
        "length": round(length, 2),
        "width": round(width, 2),
        "volume": round(volume, 2),
        "severity": severity,
        "status": status,
        "sample_count": len(raw),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Main system class
# ═══════════════════════════════════════════════════════════════════════════════

class PotholeDetectionSystem:
    """
    Orchestrates all components into a coherent detection pipeline.

    Initialise → start() → (runs until shutdown_event set) → stop()
    """

    def __init__(self, config_path: Optional[str] = None):
        # ── Configuration ──────────────────────────────────────────────────
        cfg_file = config_path or str(_HERE.parent / "config.json")
        self.cfg = SystemConfig.from_file(cfg_file)

        # ── Logging ────────────────────────────────────────────────────────
        self.log = setup_logging(self.cfg)
        self.log.info("=" * 65)
        self.log.info("  Smart Pothole Detection System  — Starting Up")
        self.log.info("  Hardware: RPi 4B + TF-02 Pro LiDAR + NEO-6M GPS")
        self.log.info("=" * 65)

        # ── Shutdown machinery ─────────────────────────────────────────────
        self._shutdown = threading.Event()
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        # ── Session stats ──────────────────────────────────────────────────
        self._start_time = time.time()
        self._detection_count = 0
        self._frame_count = 0

        # ── Hardware ───────────────────────────────────────────────────────
        self.log.info("Initialising hardware …")
        self.lidar = self._init_lidar()
        self.gps = self._init_gps()

        # ── Algorithm components ───────────────────────────────────────────
        d = self.cfg.detection
        self.depth_builder = DepthImageBuilder(
            buffer_size=d.buffer_size,
            image_width=d.depth_image_width,
            image_height=d.depth_image_height,
            sensor_height_cm=self.cfg.lidar.sensor_height_cm,
            baseline_percentile=d.baseline_percentile,
        )

        # ── YOLO detector ──────────────────────────────────────────────────
        y = self.cfg.yolo
        self.detector = YOLOPotholeDetector(
            model_path=y.model_path,
            pretrained_base=y.pretrained_base,
            confidence=y.confidence_threshold,
            iou=y.iou_threshold,
            device=y.device,
            image_size=y.input_size,
        )

        # ── Online trainer (optional) ──────────────────────────────────────
        t = self.cfg.training
        self.trainer: Optional[OnlineTrainer] = None
        if t.enabled:
            self.trainer = OnlineTrainer(
                data_dir=t.data_dir,
                model_path=t.save_model_path,
                backup_path=t.backup_model_path,
                retrain_every=t.retrain_every_n_detections,
                max_samples=t.max_samples_per_session,
                epochs=t.epochs_per_update,
                lr=t.learning_rate,
                on_model_updated=self.detector.reload_model,
            )

        # ── Backend client ─────────────────────────────────────────────────
        b = self.cfg.backend
        self.client = BackendClient(
            backend_url=b.backend_url,
            local_url=b.local_url,
            timeout_s=b.timeout_s,
            road_profile_hz=b.road_profile_upload_hz,
        )

        # ── Event tracker (state machine) ──────────────────────────────────
        self.event_tracker = PotholeEventTracker(
            threshold_cm=d.pothole_depth_threshold_cm,
            min_samples=d.min_event_samples,
            max_duration_s=d.max_event_duration_s,
            on_event=self._on_pothole_event,
        )

        self.log.info("✓ All components initialised — system ready")

    # ── Hardware init helpers ─────────────────────────────────────────────────

    def _init_lidar(self) -> Optional[TF02ProLiDAR]:
        c = self.cfg.lidar
        try:
            lidar = TF02ProLiDAR(
                port=c.port,
                baud=c.baud,
                timeout=c.timeout,
                min_cm=c.min_range_cm,
                max_cm=c.max_range_cm,
            )
            if lidar.is_ready:
                self.log.info(f"✓ TF-02 Pro LiDAR on {c.port}")
                return lidar
        except Exception as exc:
            self.log.error(f"✗ LiDAR init failed: {exc}")
        return None

    def _init_gps(self) -> Optional[NEO6MGPS]:
        c = self.cfg.gps
        try:
            gps = NEO6MGPS(
                port=c.port,
                baud=c.baud,
                fallback_ports=c.fallback_ports,
            )
            if gps.is_ready:
                self.log.info(f"✓ NEO-6M GPS — waiting for fix …")
                return gps
        except Exception as exc:
            self.log.error(f"✗ GPS init failed: {exc}")
        return None

    # ── Signal handler ────────────────────────────────────────────────────────

    def _handle_signal(self, signum, frame):
        self.log.info(f"Signal {signum} received — shutting down …")
        self._shutdown.set()

    # ── Pothole event handler ─────────────────────────────────────────────────

    def _on_pothole_event(
        self,
        readings: List[float],
        start_time: float,
        gps: dict,
    ) -> None:
        """
        Called by PotholeEventTracker when a pothole event window closes.
        Runs on the main loop thread — must be fast (no I/O blocking).
        """
        duration = time.time() - start_time
        det_cfg = self.cfg.detection

        # ── Physical measurements ─────────────────────────────────────────
        dims = calculate_pothole_dimensions(
            readings_cm=readings,
            duration_s=duration,
            vehicle_speed_cms=det_cfg.vehicle_speed_cms,
            sensor_height_cm=self.cfg.lidar.sensor_height_cm,
        )
        if not dims:
            return

        depth = dims["depth"]
        severity = dims["severity"]

        # ── YOLO verification: run inference on the event profile ─────────
        detections = self.detector.detect_from_profile(
            depth_profile=readings,
            image_width=det_cfg.depth_image_width,
            image_height=det_cfg.depth_image_height,
        )
        yolo_confirmed = len(detections) > 0
        yolo_conf = max((det.confidence for det in detections), default=0.0)

        self._detection_count += 1
        self.log.info(
            f"\n{'='*60}\n"
            f"  🚨 DETECTION #{self._detection_count} — {severity} Pothole\n"
            f"  📏 Depth: {depth:.2f} cm | Length: {dims['length']:.1f} cm | Width: {dims['width']:.1f} cm\n"
            f"  📦 Volume: {dims['volume']:.0f} cm³\n"
            f"  🤖 YOLO: {'✓ confirmed' if yolo_confirmed else '〇 heuristic only'}"
            + (f" (conf={yolo_conf:.2f})" if yolo_confirmed else "") + "\n"
            f"  📍 GPS: {'fixed' if gps.get('fixed') else 'no fix'} "
            f"@ {gps.get('lat', 0):.6f}, {gps.get('lon', 0):.6f}\n"
            f"{'='*60}"
        )

        # ── Build backend payload ─────────────────────────────────────────
        payload = {
            "latitude": gps.get("lat", 0.0),
            "longitude": gps.get("lon", 0.0),
            "depth": dims["depth"],
            "avg_depth": dims["avg_depth"],
            "length": dims["length"],
            "width": dims["width"],
            "volume": dims["volume"],
            "severity": severity,
            "status": dims["status"],
            "profile": [round(r, 2) for r in readings],
            "yolo_confirmed": yolo_confirmed,
            "yolo_confidence": round(yolo_conf, 3),
            "model_version": self.detector.model_version,
            "gps_fixed": gps.get("fixed", False),
            "speed_kmh": gps.get("speed_kmh", 0.0),
            "timestamp": datetime.now().isoformat(),
        }
        self.client.send_pothole(payload)

        # ── Register with online trainer ──────────────────────────────────
        if self.trainer:
            meta = {
                "lat": gps.get("lat"),
                "lon": gps.get("lon"),
                "severity": severity,
                "depth_cm": depth,
                "yolo_confirmed": yolo_confirmed,
                "timestamp": payload["timestamp"],
            }
            threading.Thread(
                target=self.trainer.register_detection,
                args=(readings, meta),
                daemon=True,
            ).start()

    # ── Main 100 Hz detection loop ────────────────────────────────────────────

    def run(self) -> None:
        """
        100 Hz main loop.
        Each iteration:
            1. Read latest LiDAR distance.
            2. Push to DepthImageBuilder (baseline update).
            3. Compute per-sample depth drop.
            4. Feed to PotholeEventTracker (fires callback on complete events).
            5. Add road-profile point for 3-D map.
            6. Every 64 frames: run YOLO on current depth image (opportunistic).
        """
        if not self.lidar:
            self.log.critical("LiDAR unavailable — cannot start detection loop")
            return

        self.log.info("🚀 Detection loop started @ 100 Hz")

        INTERVAL = 1.0 / self.cfg.lidar.frame_hz          # 10 ms
        YOLO_SCAN_INTERVAL = 64                             # frames between YOLO sweeps
        next_tick = time.perf_counter()

        road_z = 0.0                                        # road-profile Z accumulator

        while not self._shutdown.is_set():
            t_frame_start = time.perf_counter()

            # ── 1. Read LiDAR ─────────────────────────────────────────────
            raw_cm = self.lidar.get_distance_cm()
            if raw_cm is None:
                # No valid frame yet: maintain timing
                next_tick += INTERVAL
                sleep = next_tick - time.perf_counter()
                if sleep > 0:
                    time.sleep(sleep)
                continue

            self._frame_count += 1

            # ── 2. Push to buffer ─────────────────────────────────────────
            self.depth_builder.push(raw_cm)

            # ── 3. Compute depth drop ─────────────────────────────────────
            depth_cm = self.depth_builder.get_depth_cm(raw_cm)

            # ── 4. Event tracker ──────────────────────────────────────────
            gps = self.gps.get_location() if self.gps else {
                "lat": 0.0, "lon": 0.0, "fixed": False
            }
            self.event_tracker.feed(depth_cm, t_frame_start, gps)

            # ── 5. Road-profile telemetry ──────────────────────────────────
            road_z += self.cfg.detection.vehicle_speed_cms * INTERVAL
            self.client.add_road_point(0.0, raw_cm, road_z)

            # ── 6. Periodic YOLO sweep on full depth image ────────────────
            if self._frame_count % YOLO_SCAN_INTERVAL == 0:
                img = self.depth_builder.build_rgb_image()
                if img is not None:
                    # Run in thread so the timing loop doesn't stall
                    threading.Thread(
                        target=self._periodic_yolo_sweep,
                        args=(img,),
                        daemon=True,
                    ).start()

            # ── Precise 100 Hz timing ──────────────────────────────────────
            next_tick += INTERVAL
            sleep_s = next_tick - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            elif sleep_s < -INTERVAL * 5:
                # We've fallen behind; re-sync clock
                next_tick = time.perf_counter()

        self.log.info("Detection loop exited")
        self._print_session_summary()

    def _periodic_yolo_sweep(self, image_rgb) -> None:
        """
        Opportunistic full-image YOLO scan between event-driven detections.
        Fires on empty road too — useful for detecting slow-onset features.
        """
        detections = self.detector.detect(image_rgb)
        if detections:
            best = max(detections, key=lambda d: d.confidence)
            self.log.debug(
                f"Periodic YOLO sweep → {len(detections)} detection(s) "
                f"(best conf={best.confidence:.2f})"
            )

    # ── Shutdown & summary ─────────────────────────────────────────────────────

    def stop(self) -> None:
        """Graceful shutdown: close sensors and flush queues."""
        self.log.info("Shutting down …")
        self._shutdown.set()
        if self.lidar:
            self.lidar.close()
        if self.gps:
            self.gps.close()
        self.client.close()
        self.log.info("Shutdown complete")

    def _print_session_summary(self) -> None:
        runtime = time.time() - self._start_time
        d_stats = self.detector.get_stats()
        c_stats = self.client.get_stats()
        t_stats = self.trainer.get_stats() if self.trainer else {}
        l_stats = self.lidar.get_stats() if self.lidar else {}

        self.log.info(
            f"\n{'='*60}\n"
            f"  Session Summary\n"
            f"  Runtime       : {runtime/60:.1f} min\n"
            f"  LiDAR frames  : {self._frame_count:,}\n"
            f"  Bad checksums : {l_stats.get('bad_checksums', '—')}\n"
            f"  Potholes found: {self._detection_count}\n"
            f"  YOLO inferences: {d_stats.get('total_inferences', 0):,}\n"
            f"  Model version : {d_stats.get('model_version', 0)}\n"
            f"  Training runs : {t_stats.get('train_count', '—')}\n"
            f"  HTTP sent     : {c_stats.get('total_sent', 0)}\n"
            f"  HTTP failed   : {c_stats.get('total_failed', 0)}\n"
            f"{'='*60}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Smart Pothole Detection System (RPi 4B + TF-02 Pro + NEO-6M)"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.json (default: ../config.json relative to raspi/)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log level",
    )
    args = parser.parse_args()

    system = PotholeDetectionSystem(config_path=args.config)
    try:
        system.run()
    except KeyboardInterrupt:
        pass
    finally:
        system.stop()


if __name__ == "__main__":
    main()
