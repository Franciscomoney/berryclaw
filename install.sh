#!/bin/bash
# Berryclaw — One-command install for Raspberry Pi 5
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "🫐 Installing Berryclaw..."

# Install Python deps
pip3 install --break-system-packages -q python-telegram-bot httpx 2>/dev/null || \
pip3 install -q python-telegram-bot httpx

echo "✅ Dependencies installed"

# Create systemd user service
mkdir -p ~/.config/systemd/user

cat > ~/.config/systemd/user/berryclaw.service << EOF
[Unit]
Description=Berryclaw Telegram Bot
After=network.target

[Service]
Type=simple
WorkingDirectory=${SCRIPT_DIR}
ExecStart=/usr/bin/python3 ${SCRIPT_DIR}/berryclaw.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable berryclaw
systemctl --user start berryclaw

echo "✅ Berryclaw service started"
echo ""
echo "Useful commands:"
echo "  systemctl --user status berryclaw"
echo "  journalctl --user -u berryclaw -f"
echo "  systemctl --user restart berryclaw"
echo "  systemctl --user stop berryclaw"
