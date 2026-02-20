/* ════════════════════════════════════════════════════════════════════════════
   app.js — PotholeIQ Dashboard Logic
   WebSocket real-time updates + Leaflet map + detail panel + sparkline
   ════════════════════════════════════════════════════════════════════════════ */

'use strict';

// ─── Config ────────────────────────────────────────────────────────────────────
const API_BASE = '';                        // same origin as backend
const POLL_MS = 8000;                      // fallback polling interval
const DEFAULT_CENTER = [20.5937, 78.9629];  // India centre
const DEFAULT_ZOOM = 5;

// ─── State ─────────────────────────────────────────────────────────────────────
let allPotholes = [];                     // full data from API
let filteredList = [];                     // after filter + search
let activeFilter = 'all';
let selectedId = null;
let mapMarkers = {};                     // id → Leaflet layer
let ws = null;
let pollTimer = null;

// ─── Map Init ──────────────────────────────────────────────────────────────────
const map = L.map('map', {
    center: DEFAULT_CENTER,
    zoom: DEFAULT_ZOOM,
    zoomControl: false,
    attributionControl: true,
});

L.control.zoom({ position: 'bottomright' }).addTo(map);

L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '© OpenStreetMap © CartoDB',
    maxZoom: 19,
}).addTo(map);

// Toast container
const toastWrap = document.createElement('div');
toastWrap.className = 'toast-wrap';
document.body.appendChild(toastWrap);

// ═══════════════════════════════════════════════════════════════════════════════
//  WebSocket
// ═══════════════════════════════════════════════════════════════════════════════
function connectWS() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    ws = new WebSocket(`${proto}://${location.host}/ws/potholes`);

    ws.onopen = () => {
        setConnected(true);
        console.log('[WS] Connected');
    };

    ws.onmessage = (e) => {
        try {
            const msg = JSON.parse(e.data);
            if (msg.event === 'new_pothole' && msg.data) {
                upsertPothole(normalise(msg.data));
                renderAll();
                showToast(msg.data);
            } else if (msg.event === 'pothole_repaired') {
                const p = allPotholes.find(x => x.id === msg.id);
                if (p) { p.status = 'Green'; renderAll(); }
            }
        } catch (err) {
            console.warn('[WS] parse error', err);
        }
    };

    ws.onclose = () => {
        setConnected(false);
        console.log('[WS] Disconnected — reconnecting in 5 s');
        setTimeout(connectWS, 5000);
    };

    ws.onerror = () => ws.close();
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Data fetching
// ═══════════════════════════════════════════════════════════════════════════════
async function fetchPotholes() {
    try {
        const [ph, stats] = await Promise.all([
            fetch(`${API_BASE}/api/potholes?limit=500`).then(r => r.json()),
            fetch(`${API_BASE}/api/potholes/stats`).then(r => r.json()),
        ]);
        allPotholes = ph.map(normalise);
        updateStats(stats);
        renderAll();
        setConnected(true, 'Polling');
        document.getElementById('lastUpdate').textContent =
            'Updated ' + new Date().toLocaleTimeString();
    } catch (err) {
        console.error('[API] fetch failed', err);
        setConnected(false);
    }
}

function refreshNow() {
    clearTimeout(pollTimer);
    fetchPotholes().then(() => schedulePoll());
    // Spin the ↻ button
    const btn = document.querySelector('.conn-refresh');
    btn.style.transform = 'rotate(360deg)';
    setTimeout(() => btn.style.transform = '', 400);
}

function schedulePoll() {
    pollTimer = setTimeout(() => {
        fetchPotholes().then(() => schedulePoll());
    }, POLL_MS);
}

// ─── Normalise row from backend ────────────────────────────────────────────────
function normalise(p) {
    return {
        id: p.id,
        lat: parseFloat(p.latitude || p.lat || 0),
        lon: parseFloat(p.longitude || p.lon || 0),
        depth: parseFloat(p.depth || 0),
        avg_depth: parseFloat(p.avg_depth || 0),
        length: parseFloat(p.length || 0),
        width: parseFloat(p.width || 0),
        volume: parseFloat(p.volume || 0),
        severity: p.severity || p.severity_level || 'Minor',
        status: p.status || 'Orange',
        profile: Array.isArray(p.profile_data) ? p.profile_data : [],
        yolo_confirmed: !!p.yolo_confirmed,
        yolo_conf: parseFloat(p.yolo_confidence || 0),
        model_ver: parseInt(p.model_version || 0, 10),
        gps_fixed: !!p.gps_fixed,
        detected_at: p.detected_at || p.timestamp || '',
        repaired_at: p.repaired_at || null,
    };
}

function upsertPothole(p) {
    const idx = allPotholes.findIndex(x => x.id === p.id);
    if (idx >= 0) allPotholes[idx] = p;
    else allPotholes.unshift(p);
}

// ─── Stats panel ───────────────────────────────────────────────────────────────
function updateStats(s) {
    setText('statTotal', s.total ?? allPotholes.length);
    setText('statCritical', s.critical ?? 0);
    setText('statModerate', s.moderate ?? 0);
    setText('statRepaired', s.repaired ?? 0);
    setText('modelVersion', s.latest_model_version ?? '—');
    setText('modelYoloRate', s.yolo_confirmed != null
        ? `${s.yolo_confirmed} / ${s.total}` : '—');
    setText('modelAvgConf', s.avg_yolo_conf != null
        ? (s.avg_yolo_conf * 100).toFixed(1) + '%' : '—');
    setText('modelAvgDepth', s.avg_depth != null ? s.avg_depth + ' cm' : '—');
    setText('modelMaxDepth', s.max_depth != null ? s.max_depth + ' cm' : '—');
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Render – list + map
// ═══════════════════════════════════════════════════════════════════════════════
function renderAll() {
    applyFilter();   // sets filteredList
    renderList();
    renderMap();
}

function applyFilter() {
    const q = (document.getElementById('searchInput').value || '').toLowerCase();
    filteredList = allPotholes.filter(p => {
        if (activeFilter !== 'all' && p.status !== activeFilter) return false;
        if (q) {
            return (
                String(p.id).includes(q) ||
                p.severity.toLowerCase().includes(q) ||
                p.status.toLowerCase().includes(q)
            );
        }
        return true;
    });
}

function applySearch() {
    applyFilter();
    renderList();
    // Don't re-render map on each keystroke
}

// ── Sidebar list ───────────────────────────────────────────────────────────────
function renderList() {
    const listEl = document.getElementById('potholeList');
    const emptyEl = document.getElementById('listEmpty');

    if (filteredList.length === 0) {
        listEl.innerHTML = '';
        listEl.appendChild(emptyEl);
        emptyEl.style.display = '';
        return;
    }

    emptyEl.style.display = 'none';

    // We keep existing DOM nodes and update in-place for performance
    const existingCards = {};
    listEl.querySelectorAll('[data-id]').forEach(el => {
        existingCards[el.dataset.id] = el;
    });

    filteredList.forEach((p, i) => {
        const sid = String(p.id);
        let card = existingCards[sid];
        if (!card) {
            card = document.createElement('div');
            card.className = 'p-card';
            card.dataset.id = sid;
            card.innerHTML = `
        <div class="p-card-dot"></div>
        <div class="p-card-body">
          <div class="p-card-title"></div>
          <div class="p-card-meta"></div>
        </div>
        <div class="p-card-depth"></div>
      `;
            card.addEventListener('click', () => selectPothole(p.id));
        }

        // Update class / content
        card.className = `p-card ${colourClass(p)} ${p.id === selectedId ? 'selected' : ''}`;
        card.querySelector('.p-card-title').textContent =
            `#${p.id} — ${p.severity}${p.yolo_confirmed ? ' 🤖' : ''}`;
        card.querySelector('.p-card-meta').textContent =
            formatTime(p.detected_at) + (p.gps_fixed ? ' · 📍 GPS' : '');
        card.querySelector('.p-card-depth').textContent = p.depth.toFixed(1) + ' cm';

        listEl.appendChild(card);
        delete existingCards[sid];
    });

    // Remove stale cards
    Object.values(existingCards).forEach(el => el.remove());
}

// ── Map markers ────────────────────────────────────────────────────────────────
function renderMap() {
    const visibleIds = new Set(filteredList.map(p => p.id));

    // Remove stale markers
    Object.keys(mapMarkers).forEach(id => {
        if (!visibleIds.has(Number(id))) {
            map.removeLayer(mapMarkers[id]);
            delete mapMarkers[id];
        }
    });

    // Add / update markers
    filteredList.forEach(p => {
        if (!p.lat || !p.lon) return;

        const colour = markerColour(p);
        const radius = p.status === 'Red' ? 11 : p.status === 'Orange' ? 9 : 7;
        const popupHtml = `
      <strong>Pothole #${p.id}</strong><br/>
      ${p.severity}${p.yolo_confirmed ? ' <em>(YOLO ✓)</em>' : ''}<br/>
      ${p.depth.toFixed(1)} cm deep · ${p.length.toFixed(0)} × ${p.width.toFixed(0)} cm<br/>
      <small style="color:#6b7691">${formatTime(p.detected_at)}</small>
    `;

        if (mapMarkers[p.id]) {
            mapMarkers[p.id]
                .setLatLng([p.lat, p.lon])
                .setStyle({ color: colour, fillColor: colour, radius })
                .setPopupContent(popupHtml);
        } else {
            const marker = L.circleMarker([p.lat, p.lon], {
                color: colour,
                fillColor: colour,
                fillOpacity: 0.85,
                weight: 2,
                radius,
            }).bindPopup(popupHtml);
            marker.on('click', () => selectPothole(p.id));
            marker.addTo(map);
            mapMarkers[p.id] = marker;
        }
    });

    // If only valid GPS detections exist, fit bounds
    const valid = filteredList.filter(p => p.lat && p.lon);
    if (valid.length > 1 && filteredList.length !== allPotholes.length) {
        map.fitBounds(valid.map(p => [p.lat, p.lon]), { padding: [40, 40], maxZoom: 16 });
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Detail panel
// ═══════════════════════════════════════════════════════════════════════════════
function selectPothole(id) {
    selectedId = id;
    const p = allPotholes.find(x => x.id === id);
    if (!p) return;

    // Highlight sidebar card
    document.querySelectorAll('.p-card').forEach(el => {
        el.classList.toggle('selected', Number(el.dataset.id) === id);
    });

    // Open panel
    const panel = document.getElementById('detailPanel');
    panel.classList.add('open');

    // Title + badge
    document.getElementById('detailTitle').textContent = `Pothole #${p.id}`;
    const badge = document.getElementById('detailBadge');
    badge.textContent = p.severity + (p.yolo_confirmed ? ' — YOLO ✓' : '');
    badge.className = `detail-badge ${colourClass(p)}`;

    // Depth bar (max reference = 20 cm)
    setText('detailDepthVal', p.depth.toFixed(2) + ' cm');
    const pct = Math.min((p.depth / 20) * 100, 100).toFixed(1);
    const fill = document.getElementById('detailDepthBar');
    fill.style.width = pct + '%';
    fill.style.background = p.status === 'Red'
        ? 'linear-gradient(90deg,#f97316,#f04f5a)'
        : p.status === 'Orange'
            ? 'linear-gradient(90deg,#4f8ef7,#f97316)'
            : 'linear-gradient(90deg,#22c55e,#16a34a)';

    // Dimensions
    setText('dLength', p.length.toFixed(1));
    setText('dWidth', p.width.toFixed(1));
    setText('dVolume', p.volume.toFixed(0));
    setText('dSamples', p.profile.length || '—');

    // Meta
    const yoloText = p.yolo_confirmed
        ? `Confirmed (conf ${(p.yolo_conf * 100).toFixed(1)}%)`
        : 'Heuristic only';
    setText('dYolo', yoloText);
    setText('dCoords', `${p.lat.toFixed(6)}, ${p.lon.toFixed(6)}`);
    setText('dTime', formatTime(p.detected_at));
    setText('dStatus', p.status);
    setText('dModelVer', 'v' + p.model_ver);

    // Sparkline
    drawSparkline(p.profile);

    // Action buttons
    document.getElementById('btnRepair').style.display =
        p.status !== 'Green' ? '' : 'none';

    // Fly to on map
    if (p.lat && p.lon) {
        map.flyTo([p.lat, p.lon], 17, { animate: true, duration: 0.8 });
    }
}

function closeDetail() {
    document.getElementById('detailPanel').classList.remove('open');
    selectedId = null;
    document.querySelectorAll('.p-card.selected')
        .forEach(el => el.classList.remove('selected'));
}

// ── Sparkline ──────────────────────────────────────────────────────────────────
function drawSparkline(profile) {
    const canvas = document.getElementById('sparklineCanvas');
    const ctx = canvas.getContext('2d');
    const W = canvas.width = canvas.offsetWidth;
    const H = canvas.height = 60;

    ctx.clearRect(0, 0, W, H);

    if (!profile || profile.length < 2) {
        ctx.fillStyle = '#6b7691';
        ctx.font = '11px Inter';
        ctx.textAlign = 'center';
        ctx.fillText('No profile data', W / 2, H / 2 + 4);
        return;
    }

    const maxVal = Math.max(...profile, 1);
    const step = W / (profile.length - 1);

    // Fill gradient
    const grad = ctx.createLinearGradient(0, 0, 0, H);
    grad.addColorStop(0, 'rgba(79,142,247,0.35)');
    grad.addColorStop(1, 'rgba(79,142,247,0)');

    ctx.beginPath();
    ctx.moveTo(0, H);
    profile.forEach((v, i) => {
        const x = i * step;
        const y = H - (v / maxVal) * (H - 6);
        if (i === 0) ctx.lineTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.lineTo(W, H);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    // Line
    ctx.beginPath();
    profile.forEach((v, i) => {
        const x = i * step;
        const y = H - (v / maxVal) * (H - 6);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = '#4f8ef7';
    ctx.lineWidth = 2;
    ctx.stroke();
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Actions
// ═══════════════════════════════════════════════════════════════════════════════
async function repairCurrent() {
    if (!selectedId) return;
    const btn = document.getElementById('btnRepair');
    btn.disabled = true;
    btn.textContent = 'Updating…';
    try {
        await fetch(`${API_BASE}/api/potholes/${selectedId}/repair`, { method: 'PUT' });
        const p = allPotholes.find(x => x.id === selectedId);
        if (p) { p.status = 'Green'; }
        renderAll();
        closeDetail();
        showToast({ id: selectedId, status: 'Green', depth: 0 });
    } catch (e) {
        alert('Repair update failed: ' + e.message);
    } finally {
        btn.disabled = false;
        btn.textContent = '✓ Mark Repaired';
    }
}

async function deleteCurrent() {
    if (!selectedId) return;
    if (!confirm(`Delete pothole #${selectedId}?`)) return;
    try {
        await fetch(`${API_BASE}/api/potholes/${selectedId}`, { method: 'DELETE' });
        allPotholes = allPotholes.filter(x => x.id !== selectedId);
        if (mapMarkers[selectedId]) {
            map.removeLayer(mapMarkers[selectedId]);
            delete mapMarkers[selectedId];
        }
        closeDetail();
        renderAll();
    } catch (e) {
        alert('Delete failed: ' + e.message);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Filter buttons
// ═══════════════════════════════════════════════════════════════════════════════
document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        activeFilter = btn.dataset.filter;
        document.querySelectorAll('.filter-btn')
            .forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        renderAll();
    });
});

// ═══════════════════════════════════════════════════════════════════════════════
//  Toast notifications
// ═══════════════════════════════════════════════════════════════════════════════
function showToast(p) {
    const cls = p.status === 'Red' ? 'red' : p.status === 'Orange' ? 'orange' : 'green';
    const icon = p.status === 'Red' ? '🚨' : p.status === 'Orange' ? '⚠️' : '✅';
    const msg = p.status === 'Green'
        ? `Pothole #${p.id} marked as repaired`
        : `New ${p.depth ? p.depth.toFixed(1) + ' cm ' : ''}pothole #${p.id} detected`;

    const toast = document.createElement('div');
    toast.className = `toast ${cls}`;
    toast.textContent = `${icon} ${msg}`;
    toastWrap.appendChild(toast);
    setTimeout(() => toast.remove(), 4000);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Helpers
// ═══════════════════════════════════════════════════════════════════════════════
function colourClass(p) {
    if (p.status === 'Red') return 'red';
    if (p.status === 'Green') return 'green';
    return 'orange';
}

function markerColour(p) {
    if (p.status === 'Red') return '#f04f5a';
    if (p.status === 'Green') return '#22c55e';
    return '#f97316';
}

function formatTime(ts) {
    if (!ts) return '—';
    try {
        return new Date(ts).toLocaleString();
    } catch {
        return ts;
    }
}

function setText(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val ?? '—';
}

// ─── Connection status ────────────────────────────────────────────────────────
function setConnected(ok, label) {
    const dot = document.getElementById('connDot');
    const stat = document.getElementById('connStatus');
    dot.className = `conn-dot ${ok ? 'connected' : 'disconnected'}`;
    stat.textContent = ok ? (label || 'Live (WebSocket)') : 'Disconnected';
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Bootstrap
// ═══════════════════════════════════════════════════════════════════════════════
(function init() {
    connectWS();
    fetchPotholes().then(() => schedulePoll());
})();
