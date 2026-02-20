"""
sensors.py – Hardware drivers for TF-02 Pro LiDAR and NEO-6M GPS.

Design principles:
 • Each sensor runs a dedicated background thread so the main loop never blocks.
 • Thread-safe access via threading.Event and atomic assignments.
 • Graceful degradation: sensor unavailable → returns None, never raises.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, Optional

import serial
import pynmea2

logger = logging.getLogger("PotholeSystem.Sensors")


# ═══════════════════════════════════════════════════════════════════════════════
#  TF-02 Pro LiDAR
# ═══════════════════════════════════════════════════════════════════════════════

class TF02ProLiDAR:
    """
    Non-blocking driver for the Benewake TF-02 Pro single-point LiDAR.

    The sensor continuously streams 9-byte frames at the configured rate:
        Byte 0-1 : Header = 0x59 0x59
        Byte 2-3 : Distance (cm), little-endian uint16
        Byte 4-5 : Signal strength,  little-endian uint16
        Byte 6   : Reserved / temperature LSB
        Byte 7   : Reserved / temperature MSB
        Byte 8   : Checksum = (sum of bytes 0-7) & 0xFF

    A background thread continuously reads frames and stores the latest
    valid distance. The main loop calls get_distance_cm() which is O(1).
    """

    FRAME_LEN = 9
    HEADER = (0x59, 0x59)

    # Common UART ports on Raspberry Pi — tried in order if primary fails
    FALLBACK_PORTS = [
        "/dev/ttyAMA4",
        "/dev/serial0",
        "/dev/ttyAMA0",
        "/dev/ttyUSB0",
        "/dev/ttyS0",
    ]

    def __init__(self, port: str, baud: int = 115200, timeout: float = 1.0,
                 min_cm: float = 1.0, max_cm: float = 1200.0):
        self._port = port
        self._baud = baud
        self._timeout = timeout
        self._min_cm = min_cm
        self._max_cm = max_cm

        self._distance_cm: Optional[float] = None   # latest validated reading
        self._strength: int = 0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._serial: Optional[serial.Serial] = None
        self._thread: Optional[threading.Thread] = None

        self._total_frames = 0
        self._bad_checksums = 0
        self._out_of_range = 0

        self._connect()

    # ── private ──────────────────────────────────────────────────────────────

    def _connect(self) -> None:
        # Build list: configured port first, then fallbacks (no duplicates)
        ports_to_try = [self._port]
        for p in self.FALLBACK_PORTS:
            if p not in ports_to_try:
                ports_to_try.append(p)

        for port in ports_to_try:
            try:
                logger.info(f"  Trying LiDAR on {port} @ {self._baud} baud …")
                ser = serial.Serial(
                    port,
                    self._baud,
                    timeout=self._timeout,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                )

                # ── Probe: wait up to 2 s for TF-02 header (0x59 0x59) ────
                if self._probe_port(ser, port):
                    self._serial = ser
                    self._port = port
                    logger.info(f"✓ TF-02 Pro initialised on {port} @ {self._baud} baud")
                    self._thread = threading.Thread(
                        target=self._read_loop, name="LiDAR-Reader", daemon=True
                    )
                    self._thread.start()
                    return
                else:
                    ser.close()
                    logger.info(f"    {port}: opened OK but no LiDAR data detected")

            except serial.SerialException as exc:
                logger.debug(f"    {port}: could not open ({exc})")
            except Exception as exc:
                logger.debug(f"    {port}: unexpected error ({exc})")

        logger.error(
            f"✗ TF-02 Pro could not be found on any port: {ports_to_try}. "
            f"Check wiring: LiDAR TX → Pi RX, power 5 V."
        )

    def _probe_port(self, ser: serial.Serial, port: str) -> bool:
        """Read up to 2 s of data looking for a valid 0x59 0x59 frame header."""
        import time as _time
        deadline = _time.monotonic() + 2.0
        buf = bytearray()
        while _time.monotonic() < deadline:
            chunk = ser.read(ser.in_waiting or 1)
            if chunk:
                buf.extend(chunk)
                # Look for header
                for i in range(len(buf) - 1):
                    if buf[i] == 0x59 and buf[i + 1] == 0x59:
                        logger.info(f"    {port}: TF-02 Pro header detected ({len(buf)} bytes read)")
                        return True
        logger.debug(f"    {port}: no TF-02 header in {len(buf)} bytes")
        return False

    def _read_loop(self) -> None:
        """Background thread: continuously parse LiDAR frames."""
        ser = self._serial
        buf = bytearray()

        logger.debug("TF-02 Pro read thread started")
        while not self._stop_event.is_set():
            try:
                # Read available bytes; block briefly to avoid spinning
                chunk = ser.read(ser.in_waiting or 1)
                if not chunk:
                    continue
                buf.extend(chunk)

                # Parse all complete frames in the buffer
                while len(buf) >= self.FRAME_LEN:
                    # Search for header
                    if buf[0] == 0x59 and buf[1] == 0x59:
                        frame = buf[: self.FRAME_LEN]
                        buf = buf[self.FRAME_LEN :]
                        self._parse_frame(bytes(frame))
                    else:
                        # Discard one byte and slide
                        buf = buf[1:]

            except serial.SerialException as exc:
                logger.error(f"LiDAR serial error: {exc}; reopening in 2 s")
                time.sleep(2)
                try:
                    ser.close()
                    ser.open()
                except Exception:
                    pass
            except Exception as exc:
                logger.debug(f"LiDAR read error: {exc}")
                time.sleep(0.01)

        logger.debug("TF-02 Pro read thread stopped")

    def _parse_frame(self, frame: bytes) -> None:
        """Validate checksum and extract distance."""
        self._total_frames += 1

        # Checksum: low byte of sum of first 8 bytes
        expected_cs = sum(frame[:8]) & 0xFF
        if frame[8] != expected_cs:
            self._bad_checksums += 1
            return

        dist_cm = frame[2] + frame[3] * 256      # little-endian uint16
        strength = frame[4] + frame[5] * 256

        if dist_cm < self._min_cm or dist_cm > self._max_cm:
            self._out_of_range += 1
            return

        with self._lock:
            self._distance_cm = float(dist_cm)
            self._strength = strength

    # ── public API ────────────────────────────────────────────────────────────

    def get_distance_cm(self) -> Optional[float]:
        """
        Return the latest valid distance in centimetres (thread-safe).
        Returns None if no valid frame received yet or sensor unavailable.
        """
        with self._lock:
            return self._distance_cm

    def get_strength(self) -> int:
        """Return the latest signal strength (0–65535)."""
        with self._lock:
            return self._strength

    def get_stats(self) -> Dict[str, int]:
        return {
            "total_frames": self._total_frames,
            "bad_checksums": self._bad_checksums,
            "out_of_range": self._out_of_range,
        }

    @property
    def is_ready(self) -> bool:
        return self._serial is not None and self._serial.is_open

    def close(self) -> None:
        self._stop_event.set()
        if self._serial and self._serial.is_open:
            try:
                self._serial.close()
            except Exception:
                pass
        logger.info("TF-02 Pro sensor closed")


# ═══════════════════════════════════════════════════════════════════════════════
#  NEO-6M GPS
# ═══════════════════════════════════════════════════════════════════════════════

class NEO6MGPS:
    """
    Non-blocking driver for the u-blox NEO-6M GPS module.

    Reads standard NMEA sentences ($GNRMC / $GNGGA) via UART in a background
    thread and exposes the latest position through get_location().

    GPS fix quality:
        0 = no fix, 1 = GPS fix, 2 = DGPS fix
    """

    def __init__(self, port: str, baud: int = 9600,
                 fallback_ports: Optional[list] = None):
        self._port = port
        self._baud = baud
        self._fallback_ports = fallback_ports or []

        self._location: Dict = {
            "lat": 0.0,
            "lon": 0.0,
            "alt": 0.0,
            "speed_kmh": 0.0,
            "course": 0.0,
            "satellites": 0,
            "quality": 0,
            "fixed": False,
            "timestamp": None,
        }
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._serial: Optional[serial.Serial] = None
        self._thread: Optional[threading.Thread] = None

        self._connect()

    # ── private ──────────────────────────────────────────────────────────────

    def _connect(self) -> None:
        ports = [self._port] + self._fallback_ports
        for port in ports:
            try:
                self._serial = serial.Serial(port, self._baud, timeout=2)
                # Send UBX CFG commands to set 5 Hz update rate if desired
                # (NEO-6M supports up to 5 Hz on 9600 baud)
                logger.info(f"✓ NEO-6M GPS initialised on {port}")
                self._thread = threading.Thread(
                    target=self._read_loop, name="GPS-Reader", daemon=True
                )
                self._thread.start()
                return
            except serial.SerialException:
                continue
        logger.error("✗ NEO-6M GPS: could not open any port — GPS unavailable")

    def _read_loop(self) -> None:
        """Background thread: parse NMEA sentences."""
        ser = self._serial
        logger.debug("GPS read thread started — waiting for fix …")
        while not self._stop_event.is_set():
            try:
                line = ser.readline().decode("ascii", errors="ignore").strip()
                if not line.startswith("$"):
                    continue
                try:
                    msg = pynmea2.parse(line)
                except pynmea2.ParseError:
                    continue

                with self._lock:
                    # ── RMC sentence: lat, lon, speed, course, time ──────────
                    if isinstance(msg, pynmea2.types.talker.RMC):
                        if msg.status == "A":          # "A" = data valid
                            self._location.update({
                                "lat": float(msg.latitude or 0.0),
                                "lon": float(msg.longitude or 0.0),
                                "speed_kmh": float(msg.spd_over_grnd or 0.0) * 1.852,
                                "course": float(msg.true_course or 0.0),
                                "fixed": True,
                                "timestamp": str(msg.datetime) if msg.datetime else None,
                            })
                        else:
                            self._location["fixed"] = False

                    # ── GGA sentence: altitude, satellites, quality ──────────
                    elif isinstance(msg, pynmea2.types.talker.GGA):
                        if msg.gps_qual and int(msg.gps_qual) > 0:
                            self._location.update({
                                "lat": float(msg.latitude or 0.0),
                                "lon": float(msg.longitude or 0.0),
                                "alt": float(msg.altitude or 0.0),
                                "satellites": int(msg.num_sats or 0),
                                "quality": int(msg.gps_qual or 0),
                                "fixed": True,
                            })

            except serial.SerialException as exc:
                logger.error(f"GPS serial error: {exc}; retrying in 3 s")
                time.sleep(3)
            except Exception as exc:
                logger.debug(f"GPS parse error: {exc}")

        logger.debug("GPS read thread stopped")

    # ── public API ────────────────────────────────────────────────────────────

    def get_location(self) -> Dict:
        """Return a shallow copy of the latest location data (thread-safe)."""
        with self._lock:
            return dict(self._location)

    @property
    def is_fixed(self) -> bool:
        with self._lock:
            return self._location["fixed"]

    @property
    def is_ready(self) -> bool:
        return self._serial is not None and self._serial.is_open

    def close(self) -> None:
        self._stop_event.set()
        if self._serial and self._serial.is_open:
            try:
                self._serial.close()
            except Exception:
                pass
        logger.info("NEO-6M GPS sensor closed")
