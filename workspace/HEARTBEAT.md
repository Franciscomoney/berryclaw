# Heartbeat Tasks

These tasks run periodically (every 30 minutes) via the cloud model.
The bot checks this file and executes the instructions.

## Current Tasks

1. **Stay warm** — The local model is already kept warm by the warmup loop.
2. **Memory check** — If MEMORY.md is getting long (>50 notes), summarize old entries.

## Rules

- Only send a Telegram message if there's something genuinely useful to say
- "Nothing to report" = stay silent (return HEARTBEAT_OK)
- Never spam the user with heartbeat messages
