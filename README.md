# Berryclaw

<p align="center">
  <img src="berryclaw-mascot.png" alt="Berryclaw mascot" width="200">
</p>

The first Telegram AI personal assistant optimized for Raspberry Pi. Turn your Pi into a hybrid agent to run uncensored local models (don't tell my mom), tap into any OpenRouter LLM, and execute Claude Code in Ollama from one chat.

- **Chat** — type anything, get instant replies from a local model running on your Pi
- **Think** — type `/think` for complex questions routed to powerful cloud models (Grok, Claude, Gemini)
- **Build** — type `/build` to launch real [Claude Code](https://claude.ai/claude-code) on your Pi, controlled entirely from Telegram

Single Python file. ~30MB RAM. Runs 24/7 on a Raspberry Pi 5. Created by Francisco Cordoba Otalora

## Why this exists

We tried three other frameworks on the Pi (OpenClaw, ZeroClaw, NanoClaw). All failed because their system prompts are too complex for small models — 5,000+ tokens of instructions that 1.5B models can't follow. They parrot XML, hallucinate tools, or time out.

Berryclaw was built from scratch with one rule: **small models need tiny prompts**. The local model gets ~500 tokens of system prompt. The cloud model gets everything. Two brains, one bot.

### Recommended local models

| Model | Speed | Quality | Best for |
|-------|-------|---------|----------|
| `liquid/lfm-2.5-1.2b-thinking` | Fast | Best | Daily driver on Pi 5 |
| `huihui_ai/qwen2.5-abliterate:1.5b` | Fast | Best | Great driver on Pi 5 |
| `huihui_ai/gemma3-abliterated:1b` | Faster | Good | If you want more free RAM |
| `huihui_ai/qwen3-abliterated:0.6b` | Fastest | Basic | If you want an uncensored model |

Sweet spot: **1-1.5B quantized models**. Anything above 3B gets noticeably slow on CPU (~2-4 tok/s).

## How it works

```
You (Telegram)
  |
  |-- regular message --> Ollama (runs locally on your Pi, fast, free)
  |
  |-- /think question --> OpenRouter (cloud models: Grok, Gemini, etc.)
  |
  |-- /build ----------> Claude Code (real AI coding agent in a tmux session)
```

**Chat mode** uses a tiny local model (1-1.5B parameters) running on Ollama. It responds in 2-5 seconds, streams in real-time, knows your name, has personality, and remembers things. No internet needed.

**Think mode** sends your question to a cloud model via OpenRouter. It has full context — your identity, personality, memories, user profile — and can generate images, search the web, analyze photos, read documents, and more.

**Build mode** starts a real Claude Code session on your Pi using [Ollama cloud models](https://ollama.com/blog/cloud-models). You pick a model, Claude Code launches in the background, and every message you type in Telegram gets sent straight to it. Claude Code can read your files, write code, run commands — all from your phone. Type `/exit` when done.

## Quick Start

```bash
# Clone
git clone https://github.com/Franciscomoney/berryclaw.git
cd berryclaw

# Install & run — interactive setup walks you through everything
chmod +x install.sh
./install.sh
```

The install script installs Python deps, runs the **interactive setup wizard** (asks for your bot token, API keys, admin user ID), and creates a systemd service that auto-starts on boot.

Already have your config? Just run it directly: `python3 berryclaw.py`

Want to re-run the setup wizard? `python3 berryclaw.py --setup`

### Docker

If you prefer Docker, one command gets you Berryclaw + Ollama:

```bash
git clone https://github.com/Franciscomoney/berryclaw.git
cd berryclaw

# Create your config files
cp config.json.example config.json
cp secrets.json.example secrets.json

# Edit secrets.json with your bot token and API keys
nano secrets.json

# Launch
docker compose up -d
```

This starts both Ollama and Berryclaw. The bot auto-connects to the Ollama container — no need to install Ollama separately.

Pull your first model: `docker exec berryclaw-ollama ollama pull huihui_ai/qwen2.5-abliterate:1.5b`

Or just send `/start` to your bot — it'll offer model download buttons.

## What you need

- **Raspberry Pi 5** (8GB recommended) or any Linux machine
- **[Ollama](https://ollama.com)** installed with at least one model
- **Python 3.11+**
- **Telegram bot token** — get one from [@BotFather](https://t.me/BotFather)
- **[OpenRouter](https://openrouter.ai) API key** — for `/think`, skills, and power features
- **[Ollama cloud account](https://ollama.com)** (optional) — for `/build` mode with Claude Code

## All commands

### Everyday

| Command | What it does |
|---------|-------------|
| Just type | Chat with local AI (instant, free, private) |
| `/think <question>` | Ask a powerful cloud model |
| `/build` | Start Claude Code session (pick a cloud model) |
| `/exit` | Exit Claude Code, back to normal chat |
| `/stop` | Interrupt Claude Code without exiting Build Mode |
| `/imagine <prompt>` | Generate an image |
| `/see` | Analyze a photo (reply to an image) |
| `/search <query>` | Search the web |
| `/read` | Read and summarize a document (reply to a file) |
| `/voice` | Pick TTS voice (12 voices, buttons) |
| Send a voice note | Voice chat — transcribes, responds, speaks back |

### Skills & memory

| Command | What it does |
|---------|-------------|
| `/translate <text>` | Translate to any language |
| `/skills` | List all available skills |
| `/newskill name \| desc \| prompt` | Create a new skill |
| `/deleteskill <name>` | Delete a skill |
| `/remember <note>` | Save something to memory |
| `/memory` | View all memories |
| `/forget` | Clear all memories |
| `/profile` | View your auto-built user profile |

### Settings

| Command | What it does |
|---------|-------------|
| `/model` | Switch local model |
| `/modelx` | Switch cloud model |
| `/soul` | Change personality (presets or custom) |
| `/identity` | Edit bot identity |
| `/user` | Edit user context |
| `/api` | Manage API keys from Telegram |
| `/auth` | View/change project login credentials |
| `/claude` | View/add Build Mode rules |
| `/status` | Pi system stats |
| `/clear` | Reset conversation |
| `/start` | Guided setup with health checks |

## Build Mode (Claude Code)

The killer feature. Type `/build` and you get a real Claude Code agent running on your Pi, controlled from Telegram.

**How it works under the hood:**

1. You tap `/build` — the bot shows cloud model buttons
2. You pick a model (e.g. `minimax-m2.5:cloud`) — Claude Code starts in a tmux session
3. You type a message — it's injected into Claude Code via `tmux send-keys`
4. Claude Code does its thing (reads files, writes code, runs commands)
5. Berryclaw polls the tmux pane every 2 seconds and **streams the output to Telegram in real-time** — you see Claude thinking, searching files, writing code as it happens
6. You type `/stop` to interrupt, or `/exit` to kill the session and go back to normal chat

Commands like `/exit` and `/stop` work instantly even while Claude Code is processing (concurrent update handling).

**Setup for Build Mode:**

1. Sign in to Ollama cloud: `ollama signin` on your Pi
2. Install Claude Code: `npm install -g @anthropic-ai/claude-code` (or see [install docs](https://docs.anthropic.com/en/docs/claude-code/getting-started))
3. Add `ollama_api_key` to your `secrets.json`
4. Copy `send-to-telegram.sh` to `~/.claude/hooks/`
5. Create `~/.claude/settings.json`:

```json
{
  "hooks": {
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/home/YOUR_USER/.claude/hooks/send-to-telegram.sh"
          }
        ]
      }
    ]
  }
}
```

**Available cloud models:**

| Model | What it's good at |
|-------|-------------------|
| `minimax-m2.5:cloud` | Fast coding & productivity |
| `deepseek-v3.1:cloud` | 671B reasoning powerhouse |
| `qwen3-coder-next:cloud` | Agentic coding specialist |
| `glm-5:cloud` | 744B systems engineering |
| `glm-4.7-flash:cloud` | Fast GLM flash model |
| `qwen3.5:cloud` | 397B hybrid vision-language |
| `kimi-k2.5:cloud` | Multimodal agentic |

**Sending images to Claude Code:**

Just send a photo in Telegram while in Build Mode — drag and drop, paste, or use the camera. Berryclaw handles the rest:

1. Downloads the image to `~/projects/uploads/`
2. Auto-describes it using a vision model (Gemini Flash via OpenRouter) — layout, colors, text, sections, everything
3. Sends both the file path and the detailed description to Claude Code

This means any coding model can "see" your screenshots, even models without built-in vision. Send a screenshot of a website, a UI mockup, or a design — Claude Code gets a pixel-accurate description and can recreate it.

Add a caption to give specific instructions (e.g. send a photo with "build this as a React app").

**Project auth:**

Every web project is protected with basic HTTP auth (the Pi is on a university LAN). Default credentials: `admin` / `berryclaw`. Change them with `/auth <user> <pass>` in Telegram.

**Persistent servers:**

Projects started in Build Mode keep running after you `/exit`. Claude Code starts all servers with `setsid` so they're independent of the tmux session. To stop a server: `kill $(lsof -t -i :<port>)`.

## Group Chats

Add Berryclaw to any Telegram group. Two modes:

- **Standard groups** — bot only responds when @mentioned or replied to
- **Routed groups** — bot responds to ALL messages, with a dedicated persona (agent)

### Routed Groups (Multi-Agent)

Configure `group_routing` in `config.json` to assign a Telegram group to a specific agent persona. Messages in routed groups skip the @mention requirement — the bot reads and responds to everything.

```json
{
  "group_routing": {
    "-1003812395835": "zote",
    "-1003848357518": "oracle"
  }
}
```

Each routed group gets its own:
- **SOUL.md** — personality, objectives, communication style
- **AGENTS.md** — rules and constraints
- **Memory** — per-group conversation history

This powers specialized agents like Zote (lead generation) that live in their own Telegram group and operate autonomously.

### Per-Agent Model Selection

Assign different cloud models to different agents via `agent_models` in `config.json`:

```json
{
  "agent_models": {
    "zote": "anthropic/claude-sonnet-4-6"
  }
}
```

Agents without a specific model fall back to the global `openrouter_model`. You can also change models at runtime with `/modelx <agent>` (admin only).

### Smart Model Routing

For routed groups, Berryclaw detects **action intent** (e.g., "busca plomeros en Miami") and automatically switches to a more capable model for command execution, while using the default model for regular conversation. No manual model switching needed.

## Smart Memory

Berryclaw remembers things automatically — **per user**, so each person gets their own memory and profile:

- **Auto-capture** — after every `/think` response, key facts are extracted and saved. No manual `/remember` needed.
- **Smart recall** — before each `/think`, relevant memories are pulled in. Not the whole memory file — just what's relevant.
- **User profile** — every 20 `/think` calls, a profile is auto-built from your conversations. View with `/profile`.

Memory files are stored in `workspace/memory/<user_id>/`.

## Personality

Type `/soul` to pick a preset:
- **Friendly Assistant** — warm and encouraging
- **Sarcastic Buddy** — witty with dry humor
- **Professional** — precise and structured
- **Pirate** — arrr, matey!

Or type `/soul <your custom text>` for anything you want.

## Offline Fallback

If OpenRouter is down or your API key expires, `/think` automatically falls back to your local Ollama model. You'll see a "Cloud offline — falling back to local" message. The bot never breaks — it just uses the local brain instead.

## First Run — Model Auto-Pull

No models installed? No problem. On first startup, Berryclaw checks Ollama and:

1. **Notifies admins** via Telegram with 3 recommended models as tap-to-download buttons
2. `/start` and `/model` also show the same buttons when no models exist

Recommended models for Raspberry Pi 5:

| Model | Quality | Disk |
|-------|---------|------|
| `huihui_ai/qwen2.5-abliterate:1.5b` | Best | ~1 GB |
| `huihui_ai/gemma3-abliterated:1b` | Good & lighter | ~770 MB |
| `huihui_ai/qwen3-abliterated:0.6b` | Fastest, basic | ~380 MB |

Tap a button — Berryclaw pulls the model and sets it as your default. Ready to chat.

## Integrations

Drop a Python file in `integrations/`, add the API key via `/api`, restart. Done.

| Integration | Commands | API Key |
|-------------|----------|---------|
| Deepgram | Voice notes (auto) | `deepgram_api_key` |
| Firecrawl | `/scrape`, `/crawl` | `firecrawl_api_key` |
| Lead Scraper | `/leads` (Basic/Advanced) | `apify_api_key` |
| Apify | `/apify` | `apify_api_key` |
| Google Workspace | `/sheets`, `/docs` | `google_credentials_file` |

### Lead Generation (`/leads`)

The lead scraper is a button-driven Google Maps scraping pipeline. Two modes:

**Basic Mode** — Type `/leads`, tap **Basic**, enter a search query (e.g., "barbershops in Miami"). Gets ~100 leads written to a new Google Sheets tab.

**Advanced Mode** — Type `/leads`, tap **Advanced**, enter a business type, then a city. The bot auto-fetches every ZIP code for that city and runs batched scrapes (10 ZIPs per batch) for full coverage. Gets 500+ unique leads.

Both modes write to Google Sheets with columns: Name, Phone, Email, Website, Address, City, Rating, Reviews, Category, Maps URL. The pipeline is **fully deterministic** — no LLM touches data between Apify and Sheets.

**Enrichment** — After scraping, run `/leads enrich` to crawl business websites (via Firecrawl) and find missing email addresses.

The lead scraper also supports **natural language triggers** via the action tag system. In routed groups, saying "busca barbershops en Las Vegas" automatically triggers the scraper without typing `/leads`.

## Admin Dashboard

Berryclaw runs a web dashboard at `http://YOUR_PI_IP:7777` showing:

- Bot uptime and system stats (RAM, CPU temp, disk)
- Message counts and active users
- Current model configuration
- Active Build Mode sessions

Protected with basic auth (default: `admin` / `berryclaw`). Auto-refreshes every 30 seconds.

JSON API available at `/api/stats` for programmatic access.

Configure in `config.json`:

```json
{
  "dashboard_port": 7777,
  "dashboard_password": "berryclaw"
}
```

Set `dashboard_port` to `0` to disable.

## Config

**config.json** — settings:
```json
{
  "ollama_url": "http://localhost:11434",
  "default_model": "qwen25-pi",
  "max_history": 10,
  "stream_batch_tokens": 15,
  "warmup_interval_seconds": 240,
  "heartbeat_interval_seconds": 1800,
  "allowed_users": [],
  "admin_users": [YOUR_TELEGRAM_USER_ID],
  "openrouter_model": "x-ai/grok-4.1-fast",
  "memory_model": "liquid/lfm-2.5-1.2b-instruct:free",
  "auto_capture": true,
  "profile_frequency": 20,
  "group_routing": {},
  "agent_models": {}
}
```

**secrets.json** — API keys (never committed):
```json
{
  "telegram_bot_token": "YOUR_BOT_TOKEN",
  "openrouter_api_key": "YOUR_OPENROUTER_KEY",
  "ollama_api_key": "YOUR_OLLAMA_KEY",
  "deepgram_api_key": "",
  "firecrawl_api_key": "",
  "apify_api_key": "",
  "google_credentials_file": ""
}
```

## RAM Usage (Raspberry Pi 5, 8GB)

Berryclaw is designed to be lightweight. Here's what to expect depending on what you run:

### Base processes (always running)

| Process | RAM |
|---------|-----|
| Ollama daemon (no model loaded) | ~224 MB |
| Berryclaw (Python bot) | ~63 MB |
| **Base total** | **~287 MB** |

### Local models (loaded into RAM when chatting)

| Model | Parameters | RAM when loaded | Disk |
|-------|-----------|----------------|------|
| `huihui_ai/qwen3-abliterated:0.6b` | 0.6B | ~972 MB | 378 MB |
| `huihui_ai/gemma3-abliterated:1b` | 1B | ~1,158 MB | 769 MB |
| `huihui_ai/qwen2.5-abliterate:1.5b` | 1.5B | ~1,159 MB | 940 MB |
| 3B model (e.g. llama3.2:3b) | 3B | ~2,200 MB | 1.9 GB |
| 7B model (not recommended) | 7B | ~4,500 MB | 3.8 GB |

Ollama unloads models after 5 minutes of inactivity. Berryclaw pings the model every 4 minutes to keep it warm and avoid the 80-second cold start.

### Cloud models (Build Mode)

Cloud models (`minimax-m2.5:cloud`, `deepseek-v3.1:cloud`, etc.) run on Ollama's servers — **zero local RAM** for the model itself. You only pay for Claude Code's overhead:

| Process | RAM |
|---------|-----|
| Claude Code (Node.js, when `/build` is active) | ~336 MB |
| tmux session | ~6 MB |
| **Build Mode overhead** | **~342 MB** |

### Real-world scenarios on Pi 5 (8GB)

| Scenario | Total RAM | Free for other stuff |
|----------|-----------|---------------------|
| Chatting with 1.5B local model | ~1.4 GB | ~6.5 GB |
| Chatting + Build Mode (cloud model) | ~1.7 GB | ~6.2 GB |
| Chatting + Build Mode (local model stays warm) | ~2.0 GB | ~5.9 GB |
| 3B local model + Build Mode | ~2.8 GB | ~5.1 GB |

**Bottom line:** Even the heaviest setup (3B model + Build Mode) uses under 3 GB. You'll never run out of RAM on a Pi 5 with 8GB.

**Rule of thumb for model size:** Take the parameter count, multiply by 0.6 for Q4 quantization — that's roughly the RAM in GB. A 1.5B model uses ~0.9 GB, a 3B uses ~1.8 GB, a 7B uses ~4.2 GB.

## License

MIT
