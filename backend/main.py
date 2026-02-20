"""
backend/main.py – FastAPI backend for the Smart Pothole Detection System.

Endpoints
─────────
POST /api/potholes           – Report a new pothole (from Raspberry Pi)
GET  /api/potholes           – List all potholes (for dashboard)
PUT  /api/potholes/{id}/repair – Mark a pothole as repaired
DELETE /api/potholes/{id}    – Delete a pothole record
GET  /api/potholes/stats     – Aggregate statistics
POST /api/road-profile       – Upload road-profile point batch (3-D map)
GET  /api/road-profile       – Fetch latest road-profile points
GET  /api/health             – Health check (used by BackendClient probe)
WS   /ws/potholes            – Live WebSocket push to dashboard
GET  /                       – Serve dashboard index.html
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Smart Pothole Detection API",
    description="Backend for RPi 4B + TF-02 Pro + NEO-6M pothole detection system",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Paths ─────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
DB_FILE = str(_HERE / "pothole_system.db")
DASHBOARD_DIR = _HERE.parent / "dashboard"

# ─── WebSocket connection registry ────────────────────────────────────────────
_ws_clients: List[WebSocket] = []


# ═══════════════════════════════════════════════════════════════════════════════
#  Database
# ═══════════════════════════════════════════════════════════════════════════════

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # Better concurrent write performance
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db() -> None:
    conn = get_conn()
    cur = conn.cursor()

    cur.executescript("""
    CREATE TABLE IF NOT EXISTS potholes (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        latitude        REAL    NOT NULL,
        longitude       REAL    NOT NULL,
        depth           REAL    NOT NULL,
        avg_depth       REAL    DEFAULT 0,
        length          REAL    DEFAULT 0,
        width           REAL    DEFAULT 0,
        volume          REAL    DEFAULT 0,
        severity        TEXT    DEFAULT 'Minor',
        status          TEXT    DEFAULT 'Orange',
        profile_data    TEXT    DEFAULT '[]',
        yolo_confirmed  INTEGER DEFAULT 0,
        yolo_confidence REAL    DEFAULT 0,
        model_version   INTEGER DEFAULT 0,
        gps_fixed       INTEGER DEFAULT 0,
        speed_kmh       REAL    DEFAULT 0,
        detected_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        repaired_at     TIMESTAMP NULL,
        notes           TEXT    DEFAULT ''
    );

    CREATE TABLE IF NOT EXISTS road_profiles (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id  TEXT,
        points_json TEXT,
        created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS model_events (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        event_type  TEXT,   -- 'detection', 'retrain', 'reload'
        details     TEXT,
        created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Spatial index on lat/lon for nearby-duplicate detection
    CREATE INDEX IF NOT EXISTS idx_potholes_latlon
        ON potholes(latitude, longitude);
    CREATE INDEX IF NOT EXISTS idx_potholes_detected
        ON potholes(detected_at);
    """)

    # Online migrations for existing databases
    for migration in [
        "ALTER TABLE potholes ADD COLUMN avg_depth REAL DEFAULT 0",
        "ALTER TABLE potholes ADD COLUMN yolo_confirmed INTEGER DEFAULT 0",
        "ALTER TABLE potholes ADD COLUMN yolo_confidence REAL DEFAULT 0",
        "ALTER TABLE potholes ADD COLUMN model_version INTEGER DEFAULT 0",
        "ALTER TABLE potholes ADD COLUMN gps_fixed INTEGER DEFAULT 0",
        "ALTER TABLE potholes ADD COLUMN speed_kmh REAL DEFAULT 0",
        "ALTER TABLE potholes ADD COLUMN notes TEXT DEFAULT ''",
    ]:
        try:
            cur.execute(migration)
        except sqlite3.OperationalError:
            pass

    conn.commit()
    conn.close()


init_db()


# ═══════════════════════════════════════════════════════════════════════════════
#  Pydantic models
# ═══════════════════════════════════════════════════════════════════════════════

class PotholeIn(BaseModel):
    latitude: float
    longitude: float
    depth: float
    avg_depth: float = 0.0
    length: float = 0.0
    width: float = 0.0
    volume: float = 0.0
    severity: str = "Minor"
    status: str = "Orange"
    profile: List[float] = Field(default_factory=list)
    yolo_confirmed: bool = False
    yolo_confidence: float = 0.0
    model_version: int = 0
    gps_fixed: bool = False
    speed_kmh: float = 0.0
    timestamp: Optional[str] = None


class RoadProfileIn(BaseModel):
    session_id: str
    points: List[Dict[str, float]]


# ═══════════════════════════════════════════════════════════════════════════════
#  WebSocket helpers
# ═══════════════════════════════════════════════════════════════════════════════

async def _broadcast(data: dict) -> None:
    """Send JSON payload to all connected dashboard clients."""
    dead: List[WebSocket] = []
    for ws in _ws_clients:
        try:
            await ws.send_json(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.remove(ws)


@app.websocket("/ws/potholes")
async def ws_potholes(websocket: WebSocket):
    await websocket.accept()
    _ws_clients.append(websocket)
    try:
        while True:
            # Keep connection alive; dashboard doesn't send messages upstream
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)


# ═══════════════════════════════════════════════════════════════════════════════
#  Pothole endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.post("/api/potholes")
async def report_pothole(data: PotholeIn):
    """
    Receive a pothole report from the Raspberry Pi.

    Deduplication: if an existing pothole is within ~5 m (0.00005°),
    update its depth/severity rather than inserting a duplicate.
    """
    conn = get_conn()
    cur = conn.cursor()

    PROXIMITY = 0.00005   # ≈ 5 m at equator
    cur.execute(
        """SELECT id, depth, status FROM potholes
           WHERE ABS(latitude - ?) < ? AND ABS(longitude - ?) < ?
           ORDER BY detected_at DESC LIMIT 1""",
        (data.latitude, PROXIMITY, data.longitude, PROXIMITY),
    )
    existing = cur.fetchone()

    try:
        if existing and data.depth < 2.0:
            # Low depth at same location → mark as repaired
            cur.execute(
                "UPDATE potholes SET status='Green', repaired_at=? WHERE id=?",
                (datetime.utcnow(), existing["id"]),
            )
            conn.commit()
            conn.close()
            return {"status": "repaired", "id": existing["id"]}

        # Parse timestamp
        try:
            det_at = datetime.fromisoformat(data.timestamp) if data.timestamp else datetime.utcnow()
        except Exception:
            det_at = datetime.utcnow()

        cur.execute(
            """INSERT INTO potholes
               (latitude, longitude, depth, avg_depth, length, width, volume,
                severity, status, profile_data,
                yolo_confirmed, yolo_confidence, model_version,
                gps_fixed, speed_kmh, detected_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                data.latitude, data.longitude,
                data.depth, data.avg_depth,
                data.length, data.width, data.volume,
                data.severity,
                data.status,
                json.dumps(data.profile),
                int(data.yolo_confirmed),
                data.yolo_confidence,
                data.model_version,
                int(data.gps_fixed),
                data.speed_kmh,
                det_at,
            ),
        )
        conn.commit()
        pid = cur.lastrowid

        # Fetch full row for broadcast
        cur.execute("SELECT * FROM potholes WHERE id=?", (pid,))
        row = cur.fetchone()
        conn.close()

        row_dict = dict(row)
        await _broadcast({"event": "new_pothole", "data": row_dict})

        return {"status": "success", "id": pid}

    except Exception as exc:
        conn.close()
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/potholes")
async def list_potholes(limit: int = 500, status: Optional[str] = None):
    """Return all potholes, optionally filtered by status (Red/Orange/Green)."""
    conn = get_conn()
    cur = conn.cursor()
    if status:
        cur.execute(
            "SELECT * FROM potholes WHERE status=? ORDER BY detected_at DESC LIMIT ?",
            (status, limit),
        )
    else:
        cur.execute(
            "SELECT * FROM potholes ORDER BY detected_at DESC LIMIT ?", (limit,)
        )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    # Deserialise profile_data JSON string → list
    for r in rows:
        try:
            r["profile_data"] = json.loads(r.get("profile_data") or "[]")
        except Exception:
            r["profile_data"] = []
    return JSONResponse(content=rows)


@app.get("/api/potholes/stats")
async def pothole_stats():
    conn = get_conn()
    cur = conn.cursor()
    row = dict(cur.execute("""
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN status='Red'    THEN 1 ELSE 0 END) AS critical,
            SUM(CASE WHEN status='Orange' THEN 1 ELSE 0 END) AS moderate,
            SUM(CASE WHEN status='Green'  THEN 1 ELSE 0 END) AS repaired,
            COALESCE(SUM(yolo_confirmed), 0)  AS yolo_confirmed,
            ROUND(AVG(depth), 2)              AS avg_depth,
            ROUND(MAX(depth), 2)              AS max_depth,
            ROUND(AVG(yolo_confidence), 3)    AS avg_yolo_conf,
            COALESCE(MAX(model_version), 0)   AS latest_model_version
        FROM potholes
    """).fetchone())
    conn.close()
    return JSONResponse(content=row)


@app.put("/api/potholes/{pothole_id}/repair")
async def repair_pothole(pothole_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE potholes SET status='Green', repaired_at=? WHERE id=?",
        (datetime.utcnow(), pothole_id),
    )
    conn.commit()
    if cur.rowcount == 0:
        conn.close()
        raise HTTPException(status_code=404, detail="Pothole not found")
    conn.close()
    await _broadcast({"event": "pothole_repaired", "id": pothole_id})
    return {"status": "success", "id": pothole_id}


@app.delete("/api/potholes/{pothole_id}")
async def delete_pothole(pothole_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM potholes WHERE id=?", (pothole_id,))
    conn.commit()
    if cur.rowcount == 0:
        conn.close()
        raise HTTPException(status_code=404, detail="Not found")
    conn.close()
    return {"status": "deleted", "id": pothole_id}


@app.delete("/api/potholes")
async def delete_all_potholes():
    conn = get_conn()
    conn.execute("DELETE FROM potholes")
    conn.commit()
    conn.close()
    return {"status": "all deleted"}


# ═══════════════════════════════════════════════════════════════════════════════
#  Road-profile endpoints (3-D map telemetry)
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/road-profile")
async def upload_road_profile(data: RoadProfileIn):
    conn = get_conn()
    conn.execute(
        "INSERT INTO road_profiles (session_id, points_json) VALUES (?,?)",
        (data.session_id, json.dumps(data.points)),
    )
    conn.commit()
    conn.close()
    return {"status": "ok"}


@app.get("/api/road-profile")
async def get_road_profile(limit: int = 200):
    """Return the latest road-profile point batches merged into a flat list."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT points_json FROM road_profiles ORDER BY id DESC LIMIT ?", (limit,)
    )
    rows = cur.fetchall()
    conn.close()
    points: list = []
    for r in rows:
        try:
            pts = json.loads(r["points_json"])
            if isinstance(pts, list):
                points.extend(pts)
        except Exception:
            pass
    return JSONResponse(content=points)


# ═══════════════════════════════════════════════════════════════════════════════
#  Static file serving
# ═══════════════════════════════════════════════════════════════════════════════

if DASHBOARD_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(DASHBOARD_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_dashboard():
        return FileResponse(str(DASHBOARD_DIR / "index.html"))

    @app.get("/admin", include_in_schema=False)
    async def serve_admin():
        return FileResponse(str(DASHBOARD_DIR / "admin.html"))


# ═══════════════════════════════════════════════════════════════════════════════
#  Dev runner
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
