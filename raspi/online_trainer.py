"""
online_trainer.py – Continual / online learning engine.

Overview
────────
Every time the system confirms a new real pothole it:
  1.  Saves the raw depth profile + YOLO-format label to disk.
  2.  Increments a counter.
  3.  Once the counter reaches `retrain_every_n_detections`, spawns a
      background training job (non-blocking) that fine-tunes the YOLO model
      on all accumulated training data.
  4.  After training completes, signals the YOLODetector to hot-reload
      the updated checkpoint.

YOLO label format (one .txt per image, same stem)
──────────────────────────────────────────────────
  <class_id>  <cx>  <cy>  <w>  <h>    (all normalised 0-1)

For a simple full-image label (the entire depth image is the pothole):
  0  0.5  0.5  1.0  1.0

Training data directory layout
───────────────────────────────
  training_data/
    images/   *.png
    labels/   *.txt
    data.yaml           ← Ultralytics dataset config

The trainer keeps at most `max_samples_per_session` images, deleting
oldest ones FIFO to prevent unbounded disk growth.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger("PotholeSystem.OnlineTrainer")

# Synthesise a minimal depth image for an annotated sample
_IMG_W = 64
_IMG_H = 64


class OnlineTrainer:
    """
    Background continual-learning engine for the YOLO pothole model.

    Parameters
    ----------
    data_dir : str | Path
        Root of the on-device training dataset.
    model_path : str | Path
        Path where the fine-tuned model checkpoint will be saved.
    backup_path : str | Path
        Safe backup written before every training run.
    retrain_every : int
        Fire a training run after this many new confirmed detections.
    max_samples : int
        Hard cap on stored training images (FIFO eviction).
    epochs : int
        Fine-tuning epochs per update (keep small for fast update on Pi).
    lr : float
        Learning rate for fine-tuning.
    on_model_updated : callable | None
        Called with no args after each successful retrain so the detector
        can hot-reload the checkpoint.
    """

    # YOLO dataset config template
    _DATA_YAML_TEMPLATE = """\
path: {abs_data_dir}
train: images
val: images
nc: 1
names:
  0: pothole
"""

    def __init__(
        self,
        data_dir: str,
        model_path: str,
        backup_path: str,
        retrain_every: int = 10,
        max_samples: int = 500,
        epochs: int = 3,
        lr: float = 0.001,
        on_model_updated=None,
    ):
        self._data_dir = Path(data_dir)
        self._images_dir = self._data_dir / "images"
        self._labels_dir = self._data_dir / "labels"
        self._model_path = Path(model_path)
        self._backup_path = Path(backup_path)
        self._retrain_every = retrain_every
        self._max_samples = max_samples
        self._epochs = epochs
        self._lr = lr
        self._on_model_updated = on_model_updated or (lambda: None)

        self._images_dir.mkdir(parents=True, exist_ok=True)
        self._labels_dir.mkdir(parents=True, exist_ok=True)
        self._write_data_yaml()

        self._detection_count = 0
        self._train_count = 0
        self._is_training = False
        self._lock = threading.Lock()

        # Persist detection count across restarts
        self._counter_file = self._data_dir / "detection_count.txt"
        self._load_counter()

        logger.info(
            f"OnlineTrainer ready | retrain every {retrain_every} detections | "
            f"max {max_samples} samples | {epochs} epochs/update"
        )

    # ── public API ────────────────────────────────────────────────────────────

    def register_detection(
        self,
        depth_profile: List[float],
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Called by the main loop for every confirmed pothole.

        Saves a training sample and triggers retraining if threshold reached.

        Parameters
        ----------
        depth_profile : list[float]
            Sequence of baseline-subtracted depth values (cm) from the event.
        metadata : dict, optional
            Extra info (GPS, severity, timestamp) stored alongside the sample.
        """
        sample_id = self._save_sample(depth_profile, metadata)
        with self._lock:
            self._detection_count += 1
            count = self._detection_count
        self._save_counter()

        logger.info(
            f"📚 Training sample #{count} saved (id={sample_id}) | "
            f"next retrain at {self._retrain_every * ((count // self._retrain_every) + 1)}"
        )

        if count % self._retrain_every == 0:
            self._schedule_retrain()

    def force_retrain(self) -> None:
        """Manually trigger a retrain regardless of the detection counter."""
        self._schedule_retrain()

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "detection_count": self._detection_count,
                "train_count": self._train_count,
                "is_training": self._is_training,
                "samples_on_disk": len(list(self._images_dir.glob("*.png"))),
                "retrain_every": self._retrain_every,
            }

    # ── internal – sample management ──────────────────────────────────────────

    def _save_sample(self, depth_profile: List[float], metadata: dict = None) -> str:
        """Convert depth profile → PNG + YOLO label, return sample stem."""
        import cv2 as cv

        # Enforce max_samples FIFO eviction
        imgs = sorted(self._images_dir.glob("*.png"), key=os.path.getmtime)
        while len(imgs) >= self._max_samples:
            oldest = imgs.pop(0)
            oldest.unlink(missing_ok=True)
            lbl = self._labels_dir / (oldest.stem + ".txt")
            lbl.unlink(missing_ok=True)

        # Build depth image
        arr = np.array(depth_profile, dtype=np.float32)
        x_src = np.linspace(0, 1, max(len(arr), 1))
        x_dst = np.linspace(0, 1, _IMG_W)
        resampled = np.interp(x_dst, x_src, arr).astype(np.float32)
        mx = resampled.max()
        norm = ((resampled / mx) * 255).astype(np.uint8) if mx > 0 else np.zeros(_IMG_W, dtype=np.uint8)
        image_2d = np.tile(norm, (_IMG_H, 1))
        image_rgb = np.stack([image_2d, image_2d, image_2d], axis=-1)

        # Unique filename based on timestamp
        stem = f"pothole_{int(time.time() * 1000)}"
        img_path = self._images_dir / f"{stem}.png"
        lbl_path = self._labels_dir / f"{stem}.txt"

        cv.imwrite(str(img_path), image_rgb)

        # Full-image label (class 0, centred, fills frame)
        lbl_path.write_text("0 0.500000 0.500000 1.000000 1.000000\n")

        # Save metadata alongside
        if metadata:
            import json
            meta_path = self._images_dir / f"{stem}.json"
            meta_path.write_text(json.dumps(metadata, indent=2, default=str))

        return stem

    def _write_data_yaml(self) -> None:
        yaml_path = self._data_dir / "data.yaml"
        yaml_path.write_text(
            self._DATA_YAML_TEMPLATE.format(abs_data_dir=str(self._data_dir.resolve()))
        )

    # ── internal – counter persistence ────────────────────────────────────────

    def _load_counter(self) -> None:
        try:
            self._detection_count = int(self._counter_file.read_text().strip())
            logger.info(f"Resumed detection count: {self._detection_count}")
        except Exception:
            self._detection_count = 0

    def _save_counter(self) -> None:
        try:
            self._counter_file.write_text(str(self._detection_count))
        except Exception:
            pass

    # ── internal – training ───────────────────────────────────────────────────

    def _schedule_retrain(self) -> None:
        """Kick off a non-blocking background training thread."""
        with self._lock:
            if self._is_training:
                logger.info("Training already in progress — skipping trigger")
                return
            self._is_training = True

        t = threading.Thread(
            target=self._train_loop, name="YOLO-Trainer", daemon=True
        )
        t.start()

    def _train_loop(self) -> None:
        """Actual fine-tuning; runs in a background thread."""
        import traceback

        logger.info("🏋️  YOLO online training started …")
        t0 = time.time()
        try:
            sample_count = len(list(self._images_dir.glob("*.png")))
            if sample_count < 2:
                logger.warning(
                    f"Only {sample_count} sample(s) — need ≥ 2 to train; skipping."
                )
                return

            # Back up current model before overwriting
            if self._model_path.exists():
                shutil.copy2(str(self._model_path), str(self._backup_path))
                logger.info(f"  Backup written to {self._backup_path}")

            # Refresh data.yaml in case paths changed
            self._write_data_yaml()

            from ultralytics import YOLO

            model = YOLO(str(self._model_path))

            # Fine-tune on accumulated samples
            # batch must be ≥ 2 for YOLO; cap at 8 to keep memory low on Pi
            safe_batch = max(2, min(8, sample_count))
            model.train(
                data=str(self._data_dir / "data.yaml"),
                epochs=self._epochs,
                lr0=self._lr,
                imgsz=_IMG_H,
                batch=safe_batch,
                device="cpu",
                verbose=False,
                project=str(self._data_dir / "runs"),
                name="online_train",
                exist_ok=True,
                save=True,
                plots=False,           # Disable plots to save disk I/O on Pi
                val=False,             # Skip val pass to save time
            )

            # Move best weights to model path
            best_weights = (
                self._data_dir / "runs" / "online_train" / "weights" / "best.pt"
            )
            if best_weights.exists():
                shutil.copy2(str(best_weights), str(self._model_path))
                logger.info(f"  ✓ New weights saved to {self._model_path}")
            else:
                # Try 'last.pt' as fallback
                last_weights = best_weights.parent / "last.pt"
                if last_weights.exists():
                    shutil.copy2(str(last_weights), str(self._model_path))
                    logger.info(f"  ✓ Last weights saved (best not found)")

            elapsed = time.time() - t0
            with self._lock:
                self._train_count += 1
                count = self._train_count

            logger.info(
                f"✅ YOLO training #{count} complete in {elapsed:.1f} s | "
                f"{sample_count} samples | {self._epochs} epochs"
            )

            # Notify detector to reload
            self._on_model_updated()

        except Exception:
            logger.error(f"Training failed:\n{traceback.format_exc()}")
            # Restore backup on failure
            if self._backup_path.exists():
                shutil.copy2(str(self._backup_path), str(self._model_path))
                logger.info("  ↩ Restored backup model after failure")
        finally:
            with self._lock:
                self._is_training = False
