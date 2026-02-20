"""
backend_client.py – Async-friendly HTTP client with retry logic.

Sends pothole events and road-profile telemetry to the FastAPI backend.
All uploads are enqueued and dispatched from a single background thread
so the hot detection loop is never blocked by network I/O.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger("PotholeSystem.BackendClient")


def _make_session(retries: int = 3) -> requests.Session:
    """Create a requests.Session with automatic retry on transient errors."""
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=1.0,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["POST", "GET", "PUT"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


class BackendClient:
    """
    Thread-safe, queue-backed HTTP client.

    A single background thread drains the upload queue so the main
    detection loop is never stalled by slow networks.

    Parameters
    ----------
    backend_url : str
        Base URL of the FastAPI backend (cloud or local).
    local_url : str
        Localhost URL tried first; if reachable, preferred over backend_url.
    timeout_s : float
        Per-request timeout.
    road_profile_hz : float
        How often (per second) to flush the accumulated road-profile points.
    """

    def __init__(
        self,
        backend_url: str,
        local_url: str = "http://127.0.0.1:8000",
        timeout_s: float = 5.0,
        road_profile_hz: float = 1.0,
    ):
        self._cloud_url = backend_url
        self._local_url = local_url
        self._timeout = timeout_s
        self._road_profile_interval = 1.0 / max(road_profile_hz, 0.1)

        self._base_url = self._detect_active_backend()

        self._session = _make_session()

        # Upload queue for pothole events
        self._event_queue: queue.Queue = queue.Queue()
        # Buffered road-profile points
        self._profile_buffer: List[Dict] = []
        self._profile_lock = threading.Lock()

        self._stop_event = threading.Event()
        self._total_sent = 0
        self._total_failed = 0

        # Background worker
        self._worker = threading.Thread(
            target=self._dispatch_loop,
            name="BackendClient-Worker",
            daemon=True,
        )
        self._worker.start()

        # Road-profile uploader (separate cadence)
        self._profile_worker = threading.Thread(
            target=self._profile_loop,
            name="BackendClient-ProfileUploader",
            daemon=True,
        )
        self._profile_worker.start()

        logger.info(f"✓ BackendClient ready → {self._base_url}")

    # ── backend detection ─────────────────────────────────────────────────────

    def _detect_active_backend(self) -> str:
        """Prefer local backend (same Pi) if reachable; else use cloud URL."""
        try:
            r = requests.get(f"{self._local_url}/api/health", timeout=0.5)
            if r.status_code == 200:
                logger.info(f"  Local backend reachable → {self._local_url}")
                return self._local_url
        except Exception:
            pass
        logger.info(f"  Using cloud backend → {self._cloud_url}")
        return self._cloud_url

    # ── public API ────────────────────────────────────────────────────────────

    def send_pothole(self, data: Dict[str, Any]) -> None:
        """Enqueue a pothole event for upload (non-blocking)."""
        self._event_queue.put(("pothole", data))

    def add_road_point(self, x: float, y: float, z: float) -> None:
        """Append one road-profile 3-D point to the buffer."""
        with self._profile_lock:
            self._profile_buffer.append({"x": x, "y": y, "z": z})

    def mark_repaired(self, pothole_id: int) -> None:
        """Enqueue a repair status update."""
        self._event_queue.put(("repair", pothole_id))

    def get_stats(self) -> Dict:
        return {
            "base_url": self._base_url,
            "queue_depth": self._event_queue.qsize(),
            "total_sent": self._total_sent,
            "total_failed": self._total_failed,
        }

    def close(self) -> None:
        self._stop_event.set()
        logger.info("BackendClient stopped")

    # ── background workers ────────────────────────────────────────────────────

    def _dispatch_loop(self) -> None:
        """Drain the event queue; retry on transient failures."""
        while not self._stop_event.is_set():
            try:
                kind, payload = self._event_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if kind == "pothole":
                self._post_pothole(payload)
            elif kind == "repair":
                self._put_repair(int(payload))

    def _profile_loop(self) -> None:
        """Flush road-profile buffer on a fixed cadence."""
        import uuid
        session_id = str(uuid.uuid4())
        while not self._stop_event.is_set():
            time.sleep(self._road_profile_interval)
            with self._profile_lock:
                if not self._profile_buffer:
                    continue
                batch = list(self._profile_buffer)
                self._profile_buffer.clear()
            self._post_road_profile(session_id, batch)

    # ── HTTP helpers ──────────────────────────────────────────────────────────

    def _post_pothole(self, data: Dict) -> Optional[int]:
        url = f"{self._base_url}/api/potholes"
        try:
            resp = self._session.post(url, json=data, timeout=self._timeout)
            if resp.status_code == 200:
                pid = resp.json().get("id")
                self._total_sent += 1
                logger.info(f"✅ Pothole uploaded → ID {pid}")
                return pid
            else:
                logger.warning(
                    f"⚠️ Upload HTTP {resp.status_code}: {resp.text[:120]}"
                )
                self._total_failed += 1
        except requests.exceptions.ConnectionError:
            logger.error(f"❌ Backend unreachable: {url}")
            self._total_failed += 1
            # Re-queue for retry after a delay
            time.sleep(5)
            self._event_queue.put(("pothole", data))
        except Exception as exc:
            logger.error(f"❌ Upload error: {exc}")
            self._total_failed += 1
        return None

    def _put_repair(self, pothole_id: int) -> None:
        url = f"{self._base_url}/api/potholes/{pothole_id}/repair"
        try:
            resp = self._session.put(url, timeout=self._timeout)
            if resp.status_code == 200:
                logger.info(f"✅ Pothole {pothole_id} marked repaired")
            else:
                logger.warning(f"Repair update failed: {resp.status_code}")
        except Exception as exc:
            logger.error(f"Repair PUT error: {exc}")

    def _post_road_profile(self, session_id: str, points: List[Dict]) -> None:
        url = f"{self._base_url}/api/road-profile"
        try:
            resp = self._session.post(
                url,
                json={"session_id": session_id, "points": points},
                timeout=self._timeout,
            )
            if resp.status_code != 200:
                logger.debug(f"Road profile upload: {resp.status_code}")
        except Exception:
            pass   # Road-profile failures are non-critical; silently drop
