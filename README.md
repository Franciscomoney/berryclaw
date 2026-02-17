# Berryclaw

A Telegram bot that turns your Raspberry Pi 5 into a personal AI assistant. Three modes, one chat:

- **Chat** — type anything, get instant replies from a local model running on your Pi
- **Think** — type `/think` for complex questions routed to powerful cloud models (Grok, Claude, Gemini)
- **Build** — type `/build` to launch real [Claude Code](https://claude.ai/claude-code) on your Pi, controlled entirely from Telegram

Single Python file. ~30MB RAM. Runs 24/7 on a Raspberry Pi 5.

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

# Set up config + secrets
cp config.json.example config.json
cp secrets.json.example secrets.json
# Edit secrets.json — add your Telegram bot token + OpenRouter key
# Edit config.json — set your admin user ID

# Install & run
chmod +x install.sh
./install.sh
```

The install script installs Python deps and creates a systemd service that auto-starts on boot.

Or just run it directly: `python3 berryclaw.py`

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
| `/imagine <prompt>` | Generate an image |
| `/see` | Analyze a photo (reply to an image) |
| `/search <query>` | Search the web |
| `/read` | Read and summarize a document (reply to a file) |
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
5. When it finishes, a Stop hook reads the response from the transcript and sends it back to your Telegram
6. You type `/exit` — tmux session killed, back to normal chat

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
| `qwen3.5:cloud` | 397B hybrid vision-language |
| `kimi-k2.5:cloud` | Multimodal agentic |

## Smart Memory

Berryclaw remembers things automatically:

- **Auto-capture** — after every `/think` response, key facts are extracted and saved. No manual `/remember` needed.
- **Smart recall** — before each `/think`, relevant memories are pulled in. Not the whole memory file — just what's relevant.
- **User profile** — every 20 `/think` calls, a profile is auto-built from your conversations. View with `/profile`.

## Personality

Type `/soul` to pick a preset:
- **Friendly Assistant** — warm and encouraging
- **Sarcastic Buddy** — witty with dry humor
- **Professional** — precise and structured
- **Pirate** — arrr, matey!

Or type `/soul <your custom text>` for anything you want.

## Integrations

Drop a Python file in `integrations/`, add the API key via `/api`, restart. Done.

| Integration | Commands | API Key |
|-------------|----------|---------|
| Deepgram | Voice notes (auto) | `deepgram_api_key` |
| Firecrawl | `/scrape`, `/crawl` | `firecrawl_api_key` |
| Apify | `/apify` | `apify_api_key` |
| Google Workspace | `/sheets`, `/docs` | `google_credentials_file` |

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
  "profile_frequency": 20
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

## Recommended Local Models

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| `huihui_ai/qwen2.5-abliterate:1.5b` | 1.5B | Fast | Best for Pi |
| `huihui_ai/gemma3-abliterated:1b` | 1B | Faster | Good |
| `huihui_ai/qwen3-abliterated:0.6b` | 0.6B | Fastest | Basic |

Sweet spot: **1-1.5B quantized models**. Anything above 3B is slow on Pi CPU.

## Why this exists

We tried three other frameworks on the Pi (OpenClaw, ZeroClaw, NanoClaw). All failed because their system prompts are too complex for small models — 5,000+ tokens of instructions that 1.5B models can't follow. They parrot XML, hallucinate tools, or time out.

Berryclaw was built from scratch with one rule: **small models need tiny prompts**. The local model gets ~500 tokens of system prompt. The cloud model gets everything. Two brains, one bot.

## License

MIT
