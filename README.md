# 🍓 Berryclaw

A lightweight Telegram bot optimized for **Raspberry Pi 5 + Ollama**. Two brains: a fast local model for casual chat, and a powerful cloud model (via OpenRouter) for complex tasks.

Built after OpenClaw, NanoClaw, and ZeroClaw all failed on the Pi. This is what works.

## Why Berryclaw Exists

### The Problem

We wanted a Telegram AI bot running locally on a Raspberry Pi 5 (8GB RAM, ARM64, no GPU). We tried three existing frameworks first. All of them failed.

### What We Tried (and Why It Failed)

#### OpenClaw ❌
**What it is:** A Node.js-based AI gateway that routes Telegram messages to LLMs. Supports multiple agents, skills, tools, workspaces, heartbeats, memory.

**Why it failed on Pi:**
- System prompts are **massive** — IDENTITY.md + SOUL.md + USER.md + AGENTS.md + TOOLS.md + skill descriptions + memory + conversation history. Easily 3,000-5,000 tokens before the user even says "hi".
- Small models (0.6B-1.5B) can't handle complex system prompts. They either:
  - Parrot back the XML/tool-call syntax from the prompt instead of answering
  - Waste their entire context window on internal "thinking" tags
  - Hallucinate tool calls that don't exist
- The Node.js gateway + dependencies eat ~200MB RAM just sitting idle. On a Pi with 8GB shared with the desktop and Ollama, that's a lot.
- OpenClaw is designed for cloud models (GPT-4, Claude, Grok) with huge context windows. It's overkill for a 1.5B model with 2K context.

#### ZeroClaw ❌
**What it is:** A lighter alternative to OpenClaw, also Node.js-based.

**Why it failed on Pi:**
- Same fundamental problem: system prompt too complex for small models
- The LLM kept timing out — `FailoverError: LLM request timed out` in the logs
- Still tried to use tool-calling patterns that small models can't handle
- The daemon ran but produced garbage responses or no responses at all

#### NanoClaw ❌
**What it is:** An attempt at a minimal version.

**Why it failed on Pi:**
- Still inherited the same prompt architecture assumptions
- Small models need a fundamentally different approach, not just fewer features

### The Lessons

After three failures, the pattern was clear:

| Lesson | Detail |
|--------|--------|
| **Small models need tiny prompts** | 50 tokens max for the system prompt. Not 5,000. Every token counts when your model only has 2K context. |
| **No tool calling with small models** | They can't reliably output structured function calls. They'll parrot XML or hallucinate tools. |
| **No thinking/reasoning tags** | Models under 3B can't do chain-of-thought without wasting their entire context. |
| **Two brains > one brain** | Use the tiny local model for fast chat, escalate to a cloud model for hard stuff. Best of both worlds. |
| **Keep model warm** | Ollama unloads models after 5 minutes of inactivity. Cold start on Pi = 80 seconds. A ping every 4 minutes prevents this. |
| **Python > Node.js for this** | Simpler, less RAM (~30MB vs ~200MB), easier to deploy on Pi, async with httpx works great. |
| **Markdown workspace files** | The personality/identity/skills system from OpenClaw is brilliant — but feed the full files to the CLOUD model, and only extract a one-liner for the local model. |
| **Streaming is essential** | Small models on CPU are slow (5-15 tok/s). Without streaming, the user stares at "..." for 30 seconds. With streaming, they see it typing in real-time. |

## Architecture

```
Telegram ←→ Berryclaw (Python) ←→ Ollama (local, fast, casual)
                                 ↘ OpenRouter (cloud, powerful, /think + /skills)
```

**Single Python file** (~800 lines) + markdown workspace files. No framework bloat.

### Local Brain (Ollama)
- Handles regular messages
- System prompt: ~50 tokens ("You are Berryclaw. Be helpful and brief.")
- Streaming responses with batched Telegram message edits
- Per-user model selection
- ~5-15 tokens/second on Pi 5

### Cloud Brain (OpenRouter)
- Handles `/think` queries and all skills
- Full system prompt: IDENTITY + SOUL + USER + AGENTS + MEMORY
- Supports any model: Grok, Claude, Gemini, DeepSeek, etc.
- Per-user model selection

### Workspace (Personality & Memory)
Inspired by OpenClaw's workspace pattern, but adapted for the two-brain architecture:

```
workspace/
├── IDENTITY.md       # Name, vibe, emoji (loaded for both brains)
├── SOUL.md           # Personality, rules, communication style (cloud only)
├── USER.md           # Who the bot serves (cloud only)
├── AGENTS.md         # Session behavior protocol (cloud only)
├── HEARTBEAT.md      # Periodic background tasks (cloud only)
├── MEMORY.md         # Long-term memory — bot writes, persists across restarts
└── skills/           # Markdown skill files with trigger patterns
    ├── translate.md
    ├── code.md
    ├── summarize.md
    └── eli5.md
```

**Key insight:** The local model gets a tiny extract from IDENTITY.md. The cloud model gets everything. This way small models don't choke on prompt complexity, but the cloud model still has full personality and context.

### Skills
Skills are markdown files that inject a specialized prompt into the cloud model:

```markdown
---
name: translate
trigger: /translate
description: Translate text to any language
---

Translate the user's text. If they specify a target language, use that.
If not, translate to Spanish. Return ONLY the translation.
```

No tool calling, no function schemas, no XML. Just prompt injection. Create new skills from Telegram with `/newskill`.

## Features

| Feature | Command | Brain |
|---------|---------|-------|
| Chat | Just type | Local (Ollama) |
| Complex questions | `/think <query>` | Cloud (OpenRouter) |
| Switch local model | `/model` | — |
| Switch cloud model | `/modelx` | — |
| Skills (translate, code, etc.) | `/translate <text>` | Cloud |
| Create skill | `/newskill name \| desc \| prompt` | — |
| Delete skill | `/deleteskill <name>` | — |
| List skills | `/skills` | — |
| Save memory | `/remember <note>` | — |
| View memory | `/memory` | — |
| Clear memory | `/forget` | — |
| Edit identity | `/identity [new content]` | — |
| Edit personality | `/soul [new content]` | — |
| Edit user info | `/user [new content]` | — |
| Edit agent rules | `/agents [new content]` | — |
| Pi stats | `/status` | — |
| Clear conversation | `/clear` | — |
| Model warmup | Automatic (every 4 min) | Local |
| Heartbeat tasks | Automatic (every 30 min) | Cloud |

## Requirements

- Raspberry Pi 5 (8GB recommended) or any Linux machine
- [Ollama](https://ollama.com) with at least one model pulled
- Python 3.11+
- A Telegram bot token (from [@BotFather](https://t.me/BotFather))
- An [OpenRouter](https://openrouter.ai) API key (for cloud features)

## Quick Start

```bash
# 1. Clone
git clone https://github.com/Franciscomoney/berryclaw.git
cd berryclaw

# 2. Edit config
cp config.json.example config.json
# Add your Telegram bot token, OpenRouter key, etc.

# 3. Install & run
chmod +x install.sh
./install.sh
```

The install script installs Python deps and creates a systemd user service that auto-starts on boot.

## Config

```json
{
  "telegram_bot_token": "YOUR_BOT_TOKEN",
  "ollama_url": "http://localhost:11434",
  "default_model": "qwen25-pi",
  "max_history": 10,
  "stream_batch_tokens": 15,
  "warmup_interval_seconds": 240,
  "heartbeat_interval_seconds": 1800,
  "allowed_users": [],
  "admin_users": [YOUR_TELEGRAM_USER_ID],
  "openrouter_api_key": "YOUR_OPENROUTER_KEY",
  "openrouter_model": "x-ai/grok-4.1-fast"
}
```

- `allowed_users`: empty = everyone allowed. Add Telegram user IDs to restrict.
- `admin_users`: can use `/status`, `/forget`, `/newskill`, `/deleteskill`.

## Recommended Models for Pi 5

| Model | Size | Speed | Quality | Command |
|-------|------|-------|---------|---------|
| `huihui_ai/gemma3-abliterated:1b` | 1B | Fast | Good | `ollama pull huihui_ai/gemma3-abliterated:1b` |
| `huihui_ai/qwen2.5-abliterate:1.5b` | 1.5B | Fast | Better | `ollama pull huihui_ai/qwen2.5-abliterate:1.5b` |
| `huihui_ai/qwen3-abliterated:0.6b` | 0.6B | Fastest | Basic | `ollama pull huihui_ai/qwen3-abliterated:0.6b` |

Sweet spot: **1-1.5B quantized models**. Anything above 3B gets noticeably slow on CPU.

## License

MIT
