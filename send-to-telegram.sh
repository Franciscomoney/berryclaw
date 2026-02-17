#!/bin/bash
# Stop hook for Claude Code — sends response back to Telegram
# Triggered after every Claude Code response via ~/.claude/settings.json
# Reads the conversation transcript, extracts the last assistant response,
# and posts it to the Telegram chat that initiated the request.

PENDING_FILE="$HOME/.claude/telegram_pending"
CHAT_ID_FILE="$HOME/.claude/telegram_chat_id"
BOT_TOKEN_FILE="$HOME/raspberryclaw/secrets.json"

# Only fire if there's a pending Telegram request
if [ ! -f "$PENDING_FILE" ]; then
    exit 0
fi

# Check timestamp — expire after 600 seconds (10 min)
PENDING_TS=$(cat "$PENDING_FILE" 2>/dev/null || echo "0")
NOW=$(date +%s)
DIFF=$((NOW - PENDING_TS))
if [ "$DIFF" -gt 600 ]; then
    rm -f "$PENDING_FILE"
    exit 0
fi

# Read the hook input from stdin (JSON with transcript_path, etc.)
INPUT=$(cat)
TRANSCRIPT_PATH=$(echo "$INPUT" | jq -r '.transcript_path // empty' 2>/dev/null)

if [ -z "$TRANSCRIPT_PATH" ] || [ ! -f "$TRANSCRIPT_PATH" ]; then
    exit 0
fi

# Read chat_id
CHAT_ID=$(cat "$CHAT_ID_FILE" 2>/dev/null || echo "")
if [ -z "$CHAT_ID" ]; then
    exit 0
fi

# Read bot token from secrets.json
BOT_TOKEN=$(jq -r '.telegram_bot_token // empty' "$BOT_TOKEN_FILE" 2>/dev/null)
if [ -z "$BOT_TOKEN" ]; then
    exit 0
fi

# Find the last user message line number in the transcript
LAST_USER_LINE=$(grep -n '"type":"user"' "$TRANSCRIPT_PATH" 2>/dev/null | tail -1 | cut -d: -f1)
if [ -z "$LAST_USER_LINE" ]; then
    exit 0
fi

# Extract all assistant text after the last user message
RESPONSE=$(tail -n +"$LAST_USER_LINE" "$TRANSCRIPT_PATH" \
    | grep '"type":"assistant"' \
    | jq -rs '[.[].message.content[] | select(.type == "text") | .text] | join("\n\n")' 2>/dev/null)

if [ -z "$RESPONSE" ] || [ "$RESPONSE" = "null" ]; then
    exit 0
fi

# Create temp Python script for Telegram sending
TMPSCRIPT=$(mktemp /tmp/tg-send-XXXXXX.py)
cat > "$TMPSCRIPT" << 'PYEOF'
import sys, json, re, urllib.request, time

text = sys.stdin.read().strip()
if not text:
    sys.exit(0)

chat_id = sys.argv[1]
bot_token = sys.argv[2]

def convert_md_to_html(t):
    """Convert markdown to Telegram-safe HTML."""
    t = re.sub(r'```(\w*)\n(.*?)```', r'<pre>\2</pre>', t, flags=re.DOTALL)
    t = re.sub(r'`([^`]+)`', r'<code>\1</code>', t)
    t = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', t)
    t = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<i>\1</i>', t)
    return t

def send_message(chat_id, bot_token, text, parse_mode="HTML"):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = json.dumps({
        "chat_id": int(chat_id),
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=15)
        return True
    except Exception:
        return False

def send_plain(chat_id, bot_token, text):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = json.dumps({
        "chat_id": int(chat_id),
        "text": text,
        "disable_web_page_preview": True,
    }).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=15)
    except Exception:
        pass

# Split into chunks if too long (Telegram limit: 4096 chars)
MAX_LEN = 4000
chunks = []
if len(text) <= MAX_LEN:
    chunks = [text]
else:
    lines = text.split("\n")
    current = ""
    for line in lines:
        if len(current) + len(line) + 1 > MAX_LEN:
            if current:
                chunks.append(current)
            while len(line) > MAX_LEN:
                chunks.append(line[:MAX_LEN])
                line = line[MAX_LEN:]
            current = line
        else:
            current = current + "\n" + line if current else line
    if current:
        chunks.append(current)

for i, chunk in enumerate(chunks):
    html_chunk = convert_md_to_html(chunk)
    if not send_message(chat_id, bot_token, html_chunk, "HTML"):
        send_plain(chat_id, bot_token, chunk)
    if i < len(chunks) - 1:
        time.sleep(0.3)
PYEOF

# Pipe the response through the Python script
echo "$RESPONSE" | python3 "$TMPSCRIPT" "$CHAT_ID" "$BOT_TOKEN"
rm -f "$TMPSCRIPT"

# Clean up pending file — response sent
rm -f "$PENDING_FILE"

exit 0
