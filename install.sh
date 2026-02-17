#!/bin/bash
# Berryclaw — One-command install for Raspberry Pi 5
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/raspberryclaw.log"

echo "🫐 Installing Berryclaw..."

# Install Python deps
pip3 install --break-system-packages -q python-telegram-bot httpx 2>/dev/null || \
pip3 install -q python-telegram-bot httpx

echo "✅ Dependencies installed"

# Run interactive setup if config or secrets are missing
if [ ! -f "${SCRIPT_DIR}/secrets.json" ] || grep -q "YOUR_BOT_TOKEN" "${SCRIPT_DIR}/secrets.json" 2>/dev/null; then
    echo ""
    echo "First time? Let's set up your bot."
    echo ""

    # Copy examples if they don't exist
    [ ! -f "${SCRIPT_DIR}/config.json" ] && cp "${SCRIPT_DIR}/config.json.example" "${SCRIPT_DIR}/config.json"
    [ ! -f "${SCRIPT_DIR}/secrets.json" ] && cp "${SCRIPT_DIR}/secrets.json.example" "${SCRIPT_DIR}/secrets.json"

    python3 "${SCRIPT_DIR}/berryclaw.py" --setup
fi

# Create systemd user service
mkdir -p ~/.config/systemd/user

cat > ~/.config/systemd/user/berryclaw.service << EOF
[Unit]
Description=Berryclaw Telegram Bot
After=network.target

[Service]
Type=simple
WorkingDirectory=${SCRIPT_DIR}
ExecStart=/usr/bin/python3 -u ${SCRIPT_DIR}/berryclaw.py
StandardOutput=append:${LOG_FILE}
StandardError=append:${LOG_FILE}
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
echo "  tail -f ${LOG_FILE}"
echo "  systemctl --user restart berryclaw"
echo "  systemctl --user stop berryclaw"
echo ""
echo "Now send /start to your bot on Telegram!"
