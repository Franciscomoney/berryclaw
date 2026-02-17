#!/bin/bash
# SSH to Raspberry Pi 5 via Cloudflare Tunnel
# Usage: ./connect.sh [command]
# Update TUNNEL_URL when it changes

TUNNEL_URL="warehouse-acting-door-causing.trycloudflare.com"
USER="franciscoandsam"

if [ -z "$1" ]; then
    # Interactive session
    ssh -o ProxyCommand="cloudflared access ssh --hostname $TUNNEL_URL" \
        -o StrictHostKeyChecking=no \
        "$USER@$TUNNEL_URL"
else
    # Run a command
    ssh -o ProxyCommand="cloudflared access ssh --hostname $TUNNEL_URL" \
        -o StrictHostKeyChecking=no \
        "$USER@$TUNNEL_URL" "$@"
fi
