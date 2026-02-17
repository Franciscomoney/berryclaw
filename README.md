# 🫐 Berryclaw

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

**Single Python file** + markdown workspace files. No framework bloat.

### Local Brain (Ollama)
- Handles regular messages
- System prompt: ~500 tokens — compressed personality, user context, rules
- Streaming responses with batched Telegram message edits
- Per-user model selection
- ~5-15 tokens/second on Pi 5

### Cloud Brain (OpenRouter)
- Handles `/think` queries, skills, and power skills
- Full system prompt: IDENTITY + SOUL + USER + AGENTS + PROFILE + MEMORY
- Power skills: `/imagine`, `/see`, `/search`, `/read`, `/voice`
- Supports any model: Grok, Claude, Gemini, DeepSeek, etc.
- Per-user model selection

### Why ~500 Tokens for Local? (OpenClaw Comparison)

OpenClaw injects **5,000-15,000 tokens** per call because it targets cloud models with 100K+ context windows. That's the right approach for GPT-4 or Claude — they can digest thousands of tokens of instructions without losing focus.

Our 1-1.5B local models have 8K-32K context windows, but their real bottleneck is **attention quality** — how much instruction they can reliably follow. After testing, ~500 tokens is the sweet spot.

| | OpenClaw | Berryclaw Local | Berryclaw Cloud |
|--|---------|----------------|----------------|
| Target model | GPT-4, Claude (100K+ ctx) | 1-1.5B Ollama (8-32K ctx) | Grok, Gemini, etc. |
| System prompt | 5,000-15,000 tokens | **~500 tokens** | ~2,000-5,000 tokens |
| What's included | IDENTITY + SOUL + USER + AGENTS + TOOLS + MEMORY + skill schemas | Compressed personality, user context, rules, capabilities | Full workspace files + profile + memories |
| Per-file cap | 20,000 chars | N/A (hand-crafted) | Full file |
| Total cap | 150,000 chars | ~2,000 chars | No hard cap |
| Conversation history | Full session | Last 10 messages (~500-2,000 tokens) | Last 10 messages |
| Memory | Full MEMORY.md dump | Smart recall (keyword-triggered, ~100 tokens) | Smart recall (filtered by relevance) |
| Tool schemas | 500-3,000 tokens | None (no tool calling) | None |

**Token budget per local call:**

| Layer | Tokens |
|-------|--------|
| System prompt (personality + rules) | ~500 |
| Conversation history (10 messages) | ~500-2,000 |
| Smart recall memories (when triggered) | 0-100 |
| User's message | varies |
| **Total** | **~1,000-3,000** |

This leaves 80-90% of the model's context window free for generation. The 500-token prompt gives Berryclaw real personality — it knows who it is, who you are, and how to behave — without drowning a small model in instructions it can't follow.

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
├── PROFILE.md        # Auto-built user profile (updated every 20 /think calls)
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

### Smart Memory System

Inspired by [supermemory](https://github.com/supermemoryai/openclaw-supermemory), Berryclaw has an intelligent memory system that goes beyond simple note-taking:

| Feature | How it works |
|---------|-------------|
| **Auto-capture** | After every `/think` response, a cheap fast model (Gemini Flash) extracts key facts and saves them to MEMORY.md automatically. No manual `/remember` needed. |
| **Smart recall** | Before each `/think`, relevant memories are filtered and injected into the prompt — not the entire memory file. Keeps context focused. |
| **User profile** | Every 20 `/think` calls, a profile is auto-built from conversation history and memories. Saved to `workspace/PROFILE.md`. View with `/profile`. |

The memory model (`memory_model` in config) handles extraction and filtering. Defaults to `liquid/lfm-2.5-1.2b-instruct:free` — free tier on OpenRouter.

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
| View user profile | `/profile` | — |
| Generate images | `/imagine <prompt>` | Cloud (Gemini) |
| Analyze photos | `/see [question]` | Cloud (Gemini) |
| Web search | `/search <query>` | Cloud + Exa |
| Read documents | `/read [question]` | Cloud (Gemini) |
| Transcribe voice | `/voice [question]` | Cloud (Gemini) |
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
  "openrouter_model": "x-ai/grok-4.1-fast",
  "memory_model": "liquid/lfm-2.5-1.2b-instruct:free",
  "auto_capture": true,
  "profile_frequency": 20
}
```

- `allowed_users`: empty = everyone allowed. Add Telegram user IDs to restrict.
- `admin_users`: can use `/status`, `/forget`, `/newskill`, `/deleteskill`.
- `memory_model`: which model handles memory extraction/recall (cheap and fast recommended).
- `auto_capture`: set to `false` to disable automatic fact extraction after `/think`.
- `profile_frequency`: rebuild user profile every N `/think` calls.

## Migrating from OpenClaw / ZeroClaw

If this bot previously ran OpenClaw or ZeroClaw, Telegram caches the old command menu. After switching to Berryclaw:

1. Open Telegram **Settings > Data and Storage > Storage Usage > Clear Cache**
2. Delete the chat with the bot and start a new conversation
3. The `/` menu will now show Berryclaw's commands

This is a Telegram app cache issue — the bot API side updates instantly, but the app holds onto the old list until the cache is cleared.

## Recommended Models for Pi 5

| Model | Size | Speed | Quality | Command |
|-------|------|-------|---------|---------|
| `huihui_ai/gemma3-abliterated:1b` | 1B | Fast | Good | `ollama pull huihui_ai/gemma3-abliterated:1b` |
| `huihui_ai/qwen2.5-abliterate:1.5b` | 1.5B | Fast | Better | `ollama pull huihui_ai/qwen2.5-abliterate:1.5b` |
| `huihui_ai/qwen3-abliterated:0.6b` | 0.6B | Fastest | Basic | `ollama pull huihui_ai/qwen3-abliterated:0.6b` |

Sweet spot: **1-1.5B quantized models**. Anything above 3B gets noticeably slow on CPU.

## License

MIT
