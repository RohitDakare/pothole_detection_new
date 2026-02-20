"""
config.py – System-wide configuration for the Pothole Detection System.

Hardware: Raspberry Pi 4B + Benewake TF-02 Pro LiDAR + u-blox NEO-6M GPS
All tunable parameters are centralised here; no magic numbers in other modules.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List

# ─── Computed Path Roots ────────────────────────────────────────────────────
_HERE = Path(__file__).parent          # raspi/
_ROOT = _HERE.parent                   # pothole_detection/


@dataclass
class LiDARConfig:
    """TF-02 Pro hardware parameters."""
    port: str = "/dev/ttyAMA4"         # Hardware UART4 on Pi 4B
    baud: int = 115200                 # Factory default for TF-02 Pro
    timeout: float = 1.0              # Serial read timeout (seconds)
    max_range_cm: float = 1200.0      # TF-02 Pro spec: 12 m max
    min_range_cm: float = 10.0        # TF-02 Pro spec: 10 cm min
    sensor_height_cm: float = 20.0   # Height of LiDAR above road surface (cm)
                                      #   → subtract this from raw reading to get depth
    frame_hz: int = 100               # Sensor output rate (configurable via TFPA cmd)


@dataclass
class GPSConfig:
    """NEO-6M hardware parameters."""
    port: str = "/dev/ttyAMA0"        # Primary UART on Pi 4B (/dev/serial0)
    fallback_ports: List[str] = field(
        default_factory=lambda: [
            "/dev/serial0",
            "/dev/ttyAMA0",
            "/dev/ttyUSB0",
        ]
    )
    baud: int = 9600                  # NEO-6M default
    update_rate_hz: float = 1.0       # GPS data refresh rate
    fix_timeout_s: float = 120.0      # Max wait for GPS fix before using 0,0


@dataclass
class DetectionConfig:
    """Pothole detection algorithm parameters."""
    # LiDAR sliding-window buffer
    buffer_size: int = 200            # Number of LiDAR samples in rolling window
    depth_image_width: int = 64       # 2D depth-image cols (YOLO input)
    depth_image_height: int = 64      # 2D depth-image rows (YOLO input)

    # Baseline tracking
    baseline_window: int = 30         # Samples used to compute road baseline
    baseline_percentile: float = 20.0 # Low percentile = lowest points = road surface

    # Pothole trigger conditions
    pothole_depth_threshold_cm: float = 3.0   # Min depth drop to start event
    min_event_samples: int = 5                 # Min LiDAR hits to confirm pothole
    max_event_duration_s: float = 3.0          # Longer events = sensor lift, not hole

    # Severity classification (depth, cm)
    severity_minor_max: float = 3.0
    severity_moderate_max: float = 8.0
    # > severity_moderate_max → Critical

    # Physical estimation
    vehicle_speed_kmh: float = 20.0   # Estimated traverse speed (adjust via config file)

    @property
    def vehicle_speed_cms(self) -> float:
        return self.vehicle_speed_kmh * 100 / 3.6


@dataclass
class YOLOConfig:
    """YOLOv8 model configuration."""
    model_path: str = str(_ROOT / "models" / "yolov8_pothole.pt")
    pretrained_base: str = "yolov8n.pt"       # Bootstrap from nano if custom ckpt absent
    confidence_threshold: float = 0.40
    iou_threshold: float = 0.45
    device: str = "cpu"                        # Pi 4B has no GPU; use "cpu" or "0"
    input_size: int = 64                       # Must match depth_image dimensions above


@dataclass
class OnlineTrainingConfig:
    """Continual / online learning parameters."""
    enabled: bool = True
    data_dir: str = str(_ROOT / "training_data")      # Saved labelled samples
    retrain_every_n_detections: int = 10              # Fine-tune after N new potholes
    max_samples_per_session: int = 500                # Cap stored samples
    epochs_per_update: int = 3                        # Quick fine-tune passes
    learning_rate: float = 0.001
    save_model_path: str = str(_ROOT / "models" / "yolov8_pothole.pt")
    backup_model_path: str = str(_ROOT / "models" / "yolov8_pothole_backup.pt")


@dataclass
class BackendConfig:
    """Server / API configuration."""
    backend_url: str = "http://195.35.23.26:8000"
    local_url: str = "http://127.0.0.1:8000"   # Preferred when backend runs on same Pi
    timeout_s: float = 5.0
    max_retries: int = 3
    retry_delay_s: float = 2.0
    road_profile_upload_hz: float = 1.0         # Upload road-profile batch frequency


@dataclass
class LoggingConfig:
    """Logging parameters."""
    log_dir: str = str(_ROOT / "logs")
    log_level: str = "INFO"
    max_bytes: int = 10 * 1024 * 1024           # 10 MB
    backup_count: int = 5


@dataclass
class SystemConfig:
    """
    Top-level configuration aggregating all subgroups.

    Load from JSON:
        cfg = SystemConfig.from_file("config.json")

    Save to JSON:
        cfg.save("config.json")
    """
    lidar: LiDARConfig = field(default_factory=LiDARConfig)
    gps: GPSConfig = field(default_factory=GPSConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    training: OnlineTrainingConfig = field(default_factory=OnlineTrainingConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # ── directories are created on first load ────────────────────────────────
    def __post_init__(self):
        for d in [
            self.logging.log_dir,
            self.training.data_dir,
            str(Path(self.yolo.model_path).parent),
        ]:
            Path(d).mkdir(parents=True, exist_ok=True)

    # ── serialisation helpers ─────────────────────────────────────────────────
    @classmethod
    def from_file(cls, path: str) -> "SystemConfig":
        """Load config from a JSON file, falling back to defaults on error."""
        try:
            with open(path, "r") as fh:
                raw = json.load(fh)
            lidar = LiDARConfig(**raw.get("lidar", {}))
            gps = GPSConfig(**raw.get("gps", {}))
            detection = DetectionConfig(**raw.get("detection", {}))
            yolo = YOLOConfig(**raw.get("yolo", {}))
            training = OnlineTrainingConfig(**raw.get("training", {}))
            backend = BackendConfig(**raw.get("backend", {}))
            logging_cfg = LoggingConfig(**raw.get("logging", {}))
            return cls(lidar, gps, detection, yolo, training, backend, logging_cfg)
        except FileNotFoundError:
            return cls()
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                f"Config load error ({exc}); using defaults."
            )
            return cls()

    def save(self, path: str) -> None:
        """Persist current config as JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2)
