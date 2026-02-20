#!/usr/bin/env bash
# setup.sh – Prepare the Raspberry Pi 4B for the pothole detection system.
# Run once after cloning: bash setup.sh
# Tested on Raspberry Pi OS Bookworm (64-bit) with Python 3.11+

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

# ── 4. Python dependencies ────────────────────────────────────────────────────
info "Installing Python dependencies …"
pip install -r "$(dirname "$0")/requirements.txt"

# ── 5. Download YOLOv8n base weights ─────────────────────────────────────────
MODEL_DIR="$(dirname "$0")/../models"
mkdir -p "$MODEL_DIR"
if [ ! -f "$MODEL_DIR/yolov8n.pt" ]; then
    info "Downloading YOLOv8n base weights …"
    python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
    mv yolov8n.pt "$MODEL_DIR/" 2>/dev/null || true
fi

# ── 6. Generate default config.json ───────────────────────────────────────────
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

# ── 7. systemd service ────────────────────────────────────────────────────────
SERVICE_FILE="/etc/systemd/system/pothole-detection.service"
RASPI_DIR="$(realpath "$(dirname "$0")")"
VENV_PYTHON="$(realpath "$VENV_DIR")/bin/python3"

if [ ! -f "$SERVICE_FILE" ]; then
    info "Installing systemd service …"
    sudo tee "$SERVICE_FILE" > /dev/null << EOF
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

# ── 8. Done ───────────────────────────────────────────────────────────────────
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
