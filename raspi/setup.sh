#!/usr/bin/env bash
# setup.sh – Prepare the Raspberry Pi 4B for the pothole detection system.
# Run once after cloning: bash setup.sh
# Tested on Raspberry Pi OS Bookworm (64-bit) with Python 3.11+
#
# ── SIGILL / Illegal Instruction safety ──────────────────────────────────────
# Standard PyTorch pip wheels are compiled for x86 AVX2 or ARMv8.2+ NEON,
# which the Pi 4B's Cortex-A72 (ARMv8.0) does NOT support.  Installing them
# causes SIGILL the moment any torch C-extension is loaded.
#
# Fix: install a Pi-compatible torch build FIRST, then ultralytics (which will
# detect the already-present torch and skip downloading its own copy).
# ─────────────────────────────────────────────────────────────────────────────

set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

info "=== Pothole Detection System — Setup ==="
info "Hardware: Raspberry Pi 4B + TF-02 Pro + NEO-6M"

# ── 1. OS packages ────────────────────────────────────────────────────────────
info "Installing OS packages …"
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    python3-pip python3-venv python3-dev \
    libopencv-dev libatlas-base-dev \
    libopenblas-dev \
    git build-essential

# ── 2. Enable UART interfaces ─────────────────────────────────────────────────
info "Enabling UART interfaces …"

# Disable Bluetooth to free /dev/ttyAMA0 for GPS (NEO-6M)
if ! grep -q "dtoverlay=disable-bt" /boot/firmware/config.txt 2>/dev/null; then
    echo "dtoverlay=disable-bt" | sudo tee -a /boot/firmware/config.txt
    info "  Bluetooth overlay disabled → /dev/ttyAMA0 freed for GPS"
fi

# Enable additional UARTs for LiDAR on UART4 (GPIO 8/9)
if ! grep -q "dtoverlay=uart4" /boot/firmware/config.txt 2>/dev/null; then
    echo "dtoverlay=uart4" | sudo tee -a /boot/firmware/config.txt
    info "  UART4 overlay added → /dev/ttyAMA4 for LiDAR"
fi

# Disable serial console on /dev/ttyAMA0
sudo raspi-config nonint do_serial_hw 0
sudo raspi-config nonint do_serial_cons 1

# ── 3. Python venv ────────────────────────────────────────────────────────────
VENV_DIR="$(dirname "$0")/../venv"
if [ ! -d "$VENV_DIR" ]; then
    info "Creating Python virtual environment at $VENV_DIR …"
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel

# ── 4. Core Python dependencies (no torch yet) ────────────────────────────────
info "Installing base Python dependencies …"
pip install -r "$(dirname "$0")/requirements.txt"

# ── 5. ARM-compatible PyTorch ─────────────────────────────────────────────────
# Pi 4B = Cortex-A72 = ARMv8.0-A.  AVX2 and ARMv8.2 wheels cause SIGILL.
# Strategy (in priority order):
#   a) onnxruntime  – pure-ARM, no SIGILL risk, used by ultralytics as backend
#   b) torch from piwheels (community-built ARM wheel)
#   c) torch CPU-only from PyTorch CDN  (last resort — may still SIGILL on Pi)
#
info "Installing ARM-compatible inference runtime …"

# Always install onnxruntime — fastest ARM inference, zero SIGILL risk
pip install --extra-index-url https://www.piwheels.org/simple \
    onnxruntime || warn "  onnxruntime install failed — continuing"

# Attempt piwheels torch (ARMv8.0-safe community build)
info "Trying piwheels torch (ARMv8-safe) …"
if pip install --extra-index-url https://www.piwheels.org/simple torch 2>/dev/null; then
    info "  ✓ piwheels torch installed"
else
    warn "  piwheels torch not available, trying PyTorch CPU wheel …"
    # The --index-url CPU wheels are x86-only on most releases.
    # We attempt it but will rely on onnxruntime if it SIGILL's (the probe in
    # yolo_detector.py catches that automatically).
    pip install torch --index-url https://download.pytorch.org/whl/cpu \
        || warn "  torch install from PyTorch CDN also failed — YOLO will run in heuristic-only mode"
fi

# ── 6. ultralytics (YOLO) ────────────────────────────────────────────────────
# Install AFTER torch so pip doesn't pull an incompatible torch as a dep.
info "Installing ultralytics …"
pip install "ultralytics>=8.0" || warn "  ultralytics install failed — heuristic-only mode"

# ── 7. Download YOLOv8n base weights ─────────────────────────────────────────
MODEL_DIR="$(dirname "$0")/../models"
mkdir -p "$MODEL_DIR"
if [ ! -f "$MODEL_DIR/yolov8n.pt" ]; then
    info "Downloading YOLOv8n base weights …"
    # Run in subprocess to catch SIGILL safely (same pattern as yolo_detector.py)
    python3 -c "
import subprocess, sys
result = subprocess.run(
    [sys.executable, '-c', \"from ultralytics import YOLO; YOLO('yolov8n.pt'); print('ok')\"],
    capture_output=True, text=True, timeout=120
)
if result.returncode == 0 and 'ok' in result.stdout:
    print('  ✓ yolov8n.pt downloaded')
else:
    print('  ⚠ Could not download weights (SIGILL or network error).')
    print('  Heuristic-only mode will be used.')
    print('  stderr:', result.stderr[:200])
" && mv yolov8n.pt "$MODEL_DIR/" 2>/dev/null || true
fi

# ── 8. Generate default config.json ───────────────────────────────────────────
CONFIG="$(dirname "$0")/../config.json"
if [ ! -f "$CONFIG" ]; then
    info "Generating default config.json …"
    python3 -c "
import sys; sys.path.insert(0, '$(dirname "$0")')
from config import SystemConfig
cfg = SystemConfig()
cfg.save('$CONFIG')
print('  Written to $CONFIG')
"
fi

# ── 9. systemd service ────────────────────────────────────────────────────────
SERVICE_FILE="/etc/systemd/system/pothole-detection.service"
RASPI_DIR="$(realpath "$(dirname "$0")")"
VENV_PYTHON="$(realpath "$VENV_DIR")/bin/python3"

if [ ! -f "$SERVICE_FILE" ]; then
    info "Installing systemd service …"
    sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=Smart Pothole Detection System
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$RASPI_DIR
ExecStart=$VENV_PYTHON $RASPI_DIR/main.py
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
    sudo systemctl daemon-reload
    sudo systemctl enable pothole-detection.service
    info "  Service enabled: sudo systemctl start pothole-detection"
fi

# ── 10. Done ──────────────────────────────────────────────────────────────────
info ""
info "=== Setup Complete ==="
info "  IMPORTANT: Reboot required for UART changes to take effect."
info ""
info "  After reboot:"
info "    source venv/bin/activate"
info "    python raspi/main.py"
info "  OR"
info "    sudo systemctl start pothole-detection"
warn "  Reboot now? (y/N)"
read -r ans
if [[ "$ans" =~ ^[Yy]$ ]]; then
    sudo reboot
fi
