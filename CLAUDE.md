# Berryclaw Build Environment

You are running on a Raspberry Pi 5 via Telegram. The user is chatting with you from their phone. Keep responses concise — they're reading on a small screen.

## Project Rules

Every new project gets its own folder:

```
~/projects/<project-name>/
```

Before creating a project, check what already exists: `ls ~/projects/`

Never create files outside `~/projects/`. Never modify `~/raspberryclaw/` (that's the Telegram bot).

## Port Allocation

Use ports **3000-3099** for web servers. Before binding a port, check if it's free:

```bash
lsof -i :<port>
```

Start from 3000 and go up. If 3000 is taken, use 3001, etc.

**Ports already in use (DO NOT touch):**
- 22 — SSH
- 11434 — Ollama
- 631 — CUPS

## URLs & Access

When you start a web server, the user can access it at:

```
http://10.10.49.41:<port>
```

Always tell the user the full URL after starting a server. Example:
> "Your app is running at http://10.10.49.41:3000"

The Pi is on a university LAN. This URL works from any device on the same network.

## Environment

- **Hardware**: Raspberry Pi 5, 8GB RAM, ARM64 (aarch64), no GPU
- **OS**: Debian/Raspberry Pi OS (bookworm)
- **Python**: 3.13+ (`python3`)
- **Node.js**: 22+ (`node`, `npm`)
- **Disk**: ~95GB free on `/`
- **Ollama**: Running on localhost:11434 (for AI features if needed)

## What to Use for Web Projects

For quick prototypes, prefer:
- **Python**: `python3 -m http.server <port>` for static files, FastAPI/Flask for APIs
- **Node.js**: Vite, Next.js, or Express
- **Static HTML**: Fine for simple dashboards — just serve with Python

Always install dependencies locally in the project folder (use venvs for Python, `npm install` for Node).

## Running Servers

When starting a dev server:
1. Use `nohup` or run in the background so it persists: `nohup python3 app.py &`
2. Tell the user the URL
3. To stop: `kill $(lsof -t -i :<port>)`

For production-like persistence, create a simple systemd user service or use PM2 for Node apps.

## Constraints

- **No GPU** — don't try CUDA, use CPU-only libraries
- **ARM64** — most packages work, but check if something requires x86
- **8GB RAM** — be mindful with large dependencies. Ollama is always running (~200MB + model)
- **No root** — you run as user `franciscoandsam`. Use `sudo` only if truly needed
- **No Docker** — not installed, don't suggest it

## Communication Style

The user reads your output on Telegram (small screen). Be brief:
- Don't dump huge code blocks — write files instead
- Summarize what you did in 2-3 lines
- Always end with the URL or next step
