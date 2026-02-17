#!/usr/bin/env python3
"""Berryclaw — Lightweight Telegram bot for Raspberry Pi 5 + Ollama."""

import asyncio
import json
import logging
import os
import sqlite3
import time
from pathlib import Path

import base64
import io

import httpx
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
SECRETS_PATH = BASE_DIR / "secrets.json"
WORKSPACE_DIR = BASE_DIR / "workspace"
INTEGRATIONS_DIR = BASE_DIR / "integrations"

with open(CONFIG_PATH) as f:
    CFG = json.load(f)

# Load secrets — separate file for API keys (gitignored)
# Falls back to config.json for backwards compatibility
SECRETS: dict = {}
if SECRETS_PATH.exists():
    with open(SECRETS_PATH) as f:
        SECRETS = json.load(f)


def get_secret(key: str, default: str = "") -> str:
    """Get a secret value. Checks secrets.json first, then config.json."""
    return SECRETS.get(key) or CFG.get(key, default)


BOT_TOKEN = get_secret("telegram_bot_token")
OLLAMA_URL = CFG.get("ollama_url", "http://localhost:11434").rstrip("/")
DEFAULT_MODEL = CFG.get("default_model", "qwen25-pi")
MAX_HISTORY = CFG.get("max_history", 10)
STREAM_BATCH = CFG.get("stream_batch_tokens", 15)
WARMUP_INTERVAL = CFG.get("warmup_interval_seconds", 240)
ALLOWED_USERS: list[int] = CFG.get("allowed_users", [])
ADMIN_USERS: list[int] = CFG.get("admin_users", [])
OPENROUTER_KEY = get_secret("openrouter_api_key")
OPENROUTER_MODEL = CFG.get("openrouter_model", "x-ai/grok-4.1-fast")

CLOUD_MODELS = [
    "x-ai/grok-4.1-fast",
    "minimax/minimax-m2.5",
    "z-ai/glm-5",
    "qwen/qwen3.5-plus-02-15",
    "openrouter/aurora-alpha",
]

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO
)
log = logging.getLogger("berryclaw")

# ---------------------------------------------------------------------------
# Workspace loader — build system prompts from markdown files
# ---------------------------------------------------------------------------

def _read_workspace(filename: str) -> str:
    p = WORKSPACE_DIR / filename
    return p.read_text().strip() if p.exists() else ""


PROFILE_PATH = WORKSPACE_DIR / "PROFILE.md"


def read_profile() -> str:
    """Read user profile."""
    if PROFILE_PATH.exists():
        return PROFILE_PATH.read_text().strip()
    return ""


def write_profile(content: str):
    """Write user profile."""
    PROFILE_PATH.write_text(content.strip() + "\n")


def _extract_field(text: str, field: str) -> str:
    """Extract a **Field:** value from markdown text."""
    for line in text.splitlines():
        if line.startswith(f"**{field}:**"):
            return line.split(f"**{field}:**")[1].strip()
    return ""


def build_system_prompt_local() -> str:
    """Build a ~500 token system prompt for local Ollama models.

    Balances personality depth with small-model attention limits.
    OpenClaw injects 5,000-15,000 tokens for cloud models with 100K+ context.
    Our 1-1.5B models have 8K-32K context but weak attention — they reliably
    follow ~500 tokens of instructions. This is enough for real personality
    without drowning the model.

    Budget breakdown (~500 tokens):
      - Identity & vibe:       ~30 tokens
      - Personality & style:  ~150 tokens
      - User context:         ~80 tokens
      - Behavioral rules:    ~120 tokens
      - Capabilities:         ~70 tokens
      - Safety:               ~50 tokens
    """
    identity = _read_workspace("IDENTITY.md")
    soul = _read_workspace("SOUL.md")
    user = _read_workspace("USER.md")

    # Extract identity fields
    name = _extract_field(identity, "Name") or "Berryclaw"
    vibe = _extract_field(identity, "Vibe") or "Friendly, quick, no-nonsense."
    creature = _extract_field(identity, "Creature") or "AI assistant"

    # Build the prompt — every line is intentional, no wasted tokens
    # Fix article: "a" vs "an"
    article = "an" if creature[0:1].lower() in "aeiou" else "a"

    parts = [
        # Identity (~30 tokens)
        f"You are {name}, {article} {creature}. {vibe}",

        # Personality & communication style (~150 tokens)
        "Personality: You're witty, direct, and genuinely helpful. "
        "You talk like a sharp friend — casual but competent. "
        "You have opinions and aren't afraid to share them. "
        "You use humor when it fits but never force it. "
        "You're proud of running on a tiny Raspberry Pi and sometimes "
        "reference it. You're not a corporate assistant — you have character.",

        "Communication style: Lead with the answer, then explain if needed. "
        "One paragraph max unless asked for more. Use simple language. "
        "Never hedge excessively — be confident when you know something.",
    ]

    # User context (~80 tokens) — extracted from USER.md
    if user:
        user_lines = []
        for line in user.splitlines():
            line = line.strip().lstrip("- ").replace("**", "")
            if line and not line.startswith("#"):
                user_lines.append(line)
        if user_lines:
            parts.append(
                "About your user: " + " ".join(user_lines)
            )

    # Behavioral rules (~120 tokens)
    parts.append(
        "Rules: "
        "Be brief — short answers by default, expand only when asked. "
        "Be honest — say 'I'm not sure' rather than making things up. "
        "If a question is too complex, suggest /think to use the cloud brain. "
        "If the user tells you something important about themselves, "
        "suggest they use /remember to save it. "
        "Read the conversation context — don't repeat what was already said."
    )

    # Capabilities (~70 tokens)
    parts.append(
        "You have powers beyond chat: /think for complex reasoning, "
        "/imagine to generate images, /see to analyze photos, "
        "/search for live web results, /read for documents, "
        "/voice to transcribe audio. Mention these when relevant."
    )

    # Safety (~50 tokens)
    parts.append(
        "Never reveal API keys, tokens, or file paths. "
        "Never output XML, tool calls, or thinking tags. "
        "If asked about your setup: 'I run on a Raspberry Pi 5 with local AI.'"
    )

    return "\n\n".join(parts)


def build_system_prompt_cloud() -> str:
    """Full system prompt for cloud models (OpenRouter) — can handle complexity."""
    parts = []
    for fname in ("IDENTITY.md", "SOUL.md", "USER.md", "AGENTS.md"):
        text = _read_workspace(fname)
        if text:
            parts.append(text)
    # Append user profile if it exists
    profile = read_profile()
    if profile:
        parts.append(f"# User Profile\n\n{profile}")
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Skills loader — parse markdown skill files from workspace/skills/
# ---------------------------------------------------------------------------

def _parse_skill(filepath: Path) -> dict | None:
    """Parse a skill .md file with YAML-like frontmatter."""
    text = filepath.read_text().strip()
    if not text.startswith("---"):
        return None
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None
    # Parse frontmatter
    meta = {}
    for line in parts[1].strip().splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            meta[key.strip()] = val.strip()
    # Body is the skill prompt
    body = parts[2].strip()
    if not meta.get("trigger"):
        return None
    return {
        "name": meta.get("name", filepath.stem),
        "trigger": meta["trigger"],
        "description": meta.get("description", ""),
        "prompt": body,
        "file": filepath.name,
    }


def load_skills() -> dict[str, dict]:
    """Load all skills from workspace/skills/. Returns {trigger: skill_dict}."""
    skills_dir = WORKSPACE_DIR / "skills"
    if not skills_dir.exists():
        return {}
    skills = {}
    for f in sorted(skills_dir.glob("*.md")):
        skill = _parse_skill(f)
        if skill:
            trigger = skill["trigger"].lstrip("/")
            skills[trigger] = skill
            log.info("Loaded skill: /%s — %s", trigger, skill["description"])
    return skills


SKILLS: dict[str, dict] = load_skills()


def reload_skills():
    """Reload skills from disk."""
    global SKILLS
    SKILLS = load_skills()
    log.info("Skills reloaded: %d loaded", len(SKILLS))


# ---------------------------------------------------------------------------
# Integrations loader — auto-discover API-powered skills from integrations/
# ---------------------------------------------------------------------------

def load_integrations() -> dict[str, dict]:
    """Scan integrations/ for Python modules and load those with valid secrets."""
    if not INTEGRATIONS_DIR.is_dir():
        return {}

    loaded = {}
    import importlib.util

    for filepath in sorted(INTEGRATIONS_DIR.glob("*.py")):
        if filepath.name.startswith("_"):
            continue

        try:
            spec = importlib.util.spec_from_file_location(filepath.stem, filepath)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            name = getattr(mod, "NAME", filepath.stem)
            commands = getattr(mod, "COMMANDS", {})
            required = getattr(mod, "REQUIRED_SECRETS", [])
            handle_fn = getattr(mod, "handle", None)

            if not commands or not handle_fn:
                log.warning("Integration %s: missing COMMANDS or handle()", name)
                continue

            # Check if required secrets are configured
            missing = [s for s in required if not get_secret(s)]
            if missing:
                log.info("Integration %s: skipped (missing secrets: %s)", name, ", ".join(missing))
                continue

            for cmd, desc in commands.items():
                loaded[cmd] = {
                    "name": name,
                    "description": desc,
                    "handle": handle_fn,
                    "module": mod,
                }
            log.info("Integration %s: loaded (%s)", name, ", ".join(f"/{c}" for c in commands))

        except Exception as e:
            log.error("Integration %s: failed to load: %s", filepath.stem, e)

    return loaded


INTEGRATIONS: dict[str, dict] = load_integrations()


SYSTEM_PROMPT_LOCAL = build_system_prompt_local()
SYSTEM_PROMPT_CLOUD = build_system_prompt_cloud()

WORKSPACE_FILES = {
    "identity": "IDENTITY.md",
    "soul": "SOUL.md",
    "user": "USER.md",
    "agents": "AGENTS.md",
    "heartbeat": "HEARTBEAT.md",
}

MEMORY_PATH = WORKSPACE_DIR / "MEMORY.md"
MEMORY_MODEL = CFG.get("memory_model", "liquid/lfm-2.5-1.2b-instruct:free")
AUTO_CAPTURE = CFG.get("auto_capture", True)
PROFILE_FREQUENCY = CFG.get("profile_frequency", 20)  # Update profile every N /think calls
_think_counter: int = 0


def append_memory(note: str):
    """Append a note to MEMORY.md."""
    current = MEMORY_PATH.read_text() if MEMORY_PATH.exists() else ""
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    current += f"\n- [{timestamp}] {note}"
    MEMORY_PATH.write_text(current)


def clear_memory():
    """Reset MEMORY.md to empty."""
    MEMORY_PATH.write_text("# Berryclaw Memory\n\n(empty)\n")
    reload_prompts()


def reload_prompts():
    """Reload system prompts from workspace files."""
    global SYSTEM_PROMPT_LOCAL, SYSTEM_PROMPT_CLOUD
    SYSTEM_PROMPT_LOCAL = build_system_prompt_local()
    SYSTEM_PROMPT_CLOUD = build_system_prompt_cloud()
    log.info("Prompts reloaded. Local: %d chars", len(SYSTEM_PROMPT_LOCAL))


log.info("Local system prompt (%d chars): %s", len(SYSTEM_PROMPT_LOCAL), SYSTEM_PROMPT_LOCAL[:120])

# ---------------------------------------------------------------------------
# SQLite — conversation memory + per-user model
# ---------------------------------------------------------------------------

DB_PATH = BASE_DIR / "berryclaw.db"


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            ts REAL NOT NULL
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS user_model (
            chat_id INTEGER PRIMARY KEY,
            model TEXT NOT NULL
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS user_cloud_model (
            chat_id INTEGER PRIMARY KEY,
            model TEXT NOT NULL
        )"""
    )
    conn.commit()
    return conn


DB = _db()


def get_history(chat_id: int) -> list[dict]:
    rows = DB.execute(
        "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY ts DESC LIMIT ?",
        (chat_id, MAX_HISTORY),
    ).fetchall()
    return [{"role": r, "content": c} for r, c in reversed(rows)]


def save_message(chat_id: int, role: str, content: str):
    DB.execute(
        "INSERT INTO messages (chat_id, role, content, ts) VALUES (?, ?, ?, ?)",
        (chat_id, role, content, time.time()),
    )
    DB.commit()


def clear_history(chat_id: int):
    DB.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
    DB.commit()


def get_user_model(chat_id: int) -> str:
    row = DB.execute(
        "SELECT model FROM user_model WHERE chat_id = ?", (chat_id,)
    ).fetchone()
    return row[0] if row else DEFAULT_MODEL


def set_user_model(chat_id: int, model: str):
    DB.execute(
        "INSERT OR REPLACE INTO user_model (chat_id, model) VALUES (?, ?)",
        (chat_id, model),
    )
    DB.commit()


def get_user_cloud_model(chat_id: int) -> str:
    row = DB.execute(
        "SELECT model FROM user_cloud_model WHERE chat_id = ?", (chat_id,)
    ).fetchone()
    return row[0] if row else OPENROUTER_MODEL


def set_user_cloud_model(chat_id: int, model: str):
    DB.execute(
        "INSERT OR REPLACE INTO user_cloud_model (chat_id, model) VALUES (?, ?)",
        (chat_id, model),
    )
    DB.commit()


def prune_old_messages(days: int = 7):
    cutoff = time.time() - days * 86400
    DB.execute("DELETE FROM messages WHERE ts < ?", (cutoff,))
    DB.commit()


# ---------------------------------------------------------------------------
# Access control
# ---------------------------------------------------------------------------

def is_allowed(user_id: int) -> bool:
    if not ALLOWED_USERS:
        return True
    return user_id in ALLOWED_USERS


def is_admin(user_id: int) -> bool:
    return user_id in ADMIN_USERS


# ---------------------------------------------------------------------------
# Ollama streaming chat
# ---------------------------------------------------------------------------

async def ollama_stream(model: str, messages: list[dict]):
    """Yield text chunks from Ollama streaming API."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
        async with client.stream("POST", f"{OLLAMA_URL}/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token
                if chunk.get("done"):
                    return


async def ollama_list_models() -> list[str]:
    """Get available Ollama models."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            r.raise_for_status()
            data = r.json()
            return [m["name"] for m in data.get("models", [])]
    except Exception as e:
        log.error("Failed to list models: %s", e)
        return []


# ---------------------------------------------------------------------------
# OpenRouter — cloud escalation for /think
# ---------------------------------------------------------------------------

async def openrouter_chat(messages: list[dict], model: str | None = None) -> str:
    """Send messages to OpenRouter and return the full response."""
    if not OPENROUTER_KEY:
        return "OpenRouter API key not configured. Add openrouter_api_key to config.json."

    headers = {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model or OPENROUTER_MODEL,
        "messages": messages,
    }
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            r = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        log.error("OpenRouter error: %s", e)
        return f"Cloud model error: {e}"


async def openrouter_raw(payload: dict) -> dict:
    """Send a raw payload to OpenRouter and return the full JSON response."""
    if not OPENROUTER_KEY:
        return {"error": "OpenRouter API key not configured."}
    headers = {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            r.raise_for_status()
            return r.json()
    except Exception as e:
        log.error("OpenRouter raw error: %s", e)
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Smart Memory — auto-capture, smart recall, profile building
# ---------------------------------------------------------------------------


async def auto_capture(user_msg: str, assistant_msg: str):
    """Extract key facts from a conversation turn and save to MEMORY.md.
    Runs in background after /think responses. Uses a cheap fast model."""
    try:
        messages = [
            {"role": "system", "content": (
                "You are a memory extraction system. Given a conversation between a user and an AI, "
                "extract ONLY key facts worth remembering long-term. These are facts about:\n"
                "- The user (name, preferences, projects, skills, goals)\n"
                "- Decisions made\n"
                "- Important information shared\n\n"
                "Rules:\n"
                "- Return one fact per line, starting with '- '\n"
                "- Be concise (max 15 words per fact)\n"
                "- Skip small talk, greetings, and generic questions\n"
                "- If there's nothing worth remembering, respond with exactly: NOTHING\n"
                "- Never duplicate facts already in existing memory"
            )},
            {"role": "user", "content": (
                f"Existing memory:\n{_read_workspace('MEMORY.md')}\n\n"
                f"---\n\nNew conversation:\n"
                f"User: {user_msg}\n"
                f"Assistant: {assistant_msg[:1000]}"
            )},
        ]
        result = await openrouter_chat(messages, model=MEMORY_MODEL)
        if result and "NOTHING" not in result.upper():
            # Append each extracted fact
            for line in result.strip().splitlines():
                line = line.strip()
                if line.startswith("- "):
                    append_memory(line[2:])
            log.info("Auto-capture: saved facts from conversation")
    except Exception as e:
        log.warning("Auto-capture failed: %s", e)


async def smart_recall(query: str) -> str:
    """Given a user query, return only the relevant memories from MEMORY.md.
    Uses a cheap fast model to filter. Returns empty string if no relevant memories."""
    memory = _read_workspace("MEMORY.md")
    if not memory or "(empty)" in memory:
        return ""

    # If memory is short, just return all of it
    lines = [l for l in memory.splitlines() if l.strip().startswith("- ")]
    if len(lines) <= 5:
        return memory

    try:
        messages = [
            {"role": "system", "content": (
                "You are a memory retrieval system. Given a user's question and a list of memories, "
                "return ONLY the memories that are relevant to answering the question.\n\n"
                "Rules:\n"
                "- Copy relevant memories exactly as they are (keep the '- ' prefix)\n"
                "- If no memories are relevant, respond with exactly: NONE\n"
                "- Be selective — only include memories that would actually help answer the question\n"
                "- Maximum 10 memories"
            )},
            {"role": "user", "content": (
                f"Question: {query}\n\n"
                f"All memories:\n{memory}"
            )},
        ]
        result = await openrouter_chat(messages, model=MEMORY_MODEL)
        if result and "NONE" not in result.upper():
            return result.strip()
        return ""
    except Exception as e:
        log.warning("Smart recall failed, using full memory: %s", e)
        return memory


async def update_profile(chat_id: int):
    """Rebuild user profile from conversation history and memory.
    Called every PROFILE_FREQUENCY /think calls."""
    try:
        history = get_history(chat_id)
        memory = _read_workspace("MEMORY.md")
        current_profile = read_profile()

        recent_msgs = "\n".join(
            f"{m['role'].title()}: {m['content'][:200]}" for m in history[-20:]
        )

        messages = [
            {"role": "system", "content": (
                "You are a profile builder. Given conversation history, memories, and an existing profile, "
                "create an updated user profile.\n\n"
                "The profile should contain:\n"
                "- Name and basic info\n"
                "- Interests and skills\n"
                "- Current projects\n"
                "- Preferences (language, communication style, etc.)\n"
                "- Goals\n\n"
                "Rules:\n"
                "- Keep it under 20 lines\n"
                "- Use bullet points\n"
                "- Merge new info with existing profile (don't lose old facts)\n"
                "- If nothing new to add, return the existing profile unchanged"
            )},
            {"role": "user", "content": (
                f"Existing profile:\n{current_profile or '(empty)'}\n\n"
                f"Recent memories:\n{memory}\n\n"
                f"Recent conversation:\n{recent_msgs}"
            )},
        ]
        result = await openrouter_chat(messages, model=MEMORY_MODEL)
        if result and len(result) > 20:
            write_profile(result)
            reload_prompts()
            log.info("Profile updated (%d chars)", len(result))
    except Exception as e:
        log.warning("Profile update failed: %s", e)


# ---------------------------------------------------------------------------
# Warmup — keep model loaded in memory
# ---------------------------------------------------------------------------

_last_used_model: str = DEFAULT_MODEL
HEARTBEAT_INTERVAL = CFG.get("heartbeat_interval_seconds", 1800)  # 30 min
_bot_app = None  # Set in main() for heartbeat messaging


async def warmup_loop():
    """Ping Ollama every WARMUP_INTERVAL seconds to prevent cold starts."""
    if WARMUP_INTERVAL <= 0:
        log.info("Warmup disabled")
        return
    log.info("Warmup loop started (every %ds)", WARMUP_INTERVAL)
    while True:
        await asyncio.sleep(WARMUP_INTERVAL)
        try:
            payload = {
                "model": _last_used_model,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            }
            async with httpx.AsyncClient(timeout=30.0) as client:
                await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
            log.debug("Warmup ping sent for %s", _last_used_model)
        except Exception as e:
            log.warning("Warmup failed: %s", e)


async def heartbeat_loop():
    """Run periodic heartbeat tasks via cloud model."""
    if HEARTBEAT_INTERVAL <= 0:
        log.info("Heartbeat disabled")
        return
    log.info("Heartbeat loop started (every %ds)", HEARTBEAT_INTERVAL)
    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL)
        try:
            heartbeat_prompt = _read_workspace("HEARTBEAT.md")
            if not heartbeat_prompt:
                continue

            memory = _read_workspace("MEMORY.md")
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_CLOUD},
                {"role": "user", "content": (
                    f"HEARTBEAT CHECK. Current time: {time.strftime('%Y-%m-%d %H:%M')}.\n\n"
                    f"{heartbeat_prompt}\n\n"
                    f"Current memory:\n{memory}\n\n"
                    "If you have something useful to report, respond with it. "
                    "If nothing to report, respond with exactly: HEARTBEAT_OK"
                )},
            ]
            response = await openrouter_chat(messages)
            if response and "HEARTBEAT_OK" not in response:
                # Send to admin users
                if _bot_app and ADMIN_USERS:
                    for admin_id in ADMIN_USERS:
                        try:
                            await _bot_app.bot.send_message(
                                admin_id,
                                f"💓 *Heartbeat*\n\n{response}",
                                parse_mode="Markdown",
                            )
                        except Exception:
                            pass
                log.info("Heartbeat sent message to admins")
            else:
                log.debug("Heartbeat: nothing to report")
        except Exception as e:
            log.warning("Heartbeat error: %s", e)


# ---------------------------------------------------------------------------
# Telegram handlers
# ---------------------------------------------------------------------------

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return

    # --- Health checks ---
    checks: list[str] = []
    issues: list[str] = []

    # 1. Ollama running?
    models = await ollama_list_models()
    if models:
        checks.append("✅ Ollama is running")
    else:
        issues.append("❌ Ollama is not reachable")
        issues.append("   → Make sure Ollama is installed and running: `ollama serve`")

    # 2. Models available?
    if models:
        current = get_user_model(update.effective_chat.id)
        checks.append(f"✅ {len(models)} model{'s' if len(models) != 1 else ''} available — using `{current}`")
    elif not issues:  # Ollama up but 0 models
        issues.append("⚠️ No models installed yet")
        issues.append("   → Tap the button below to pull one, or run: `ollama pull qwen2.5:1.5b`")

    # 3. OpenRouter key?
    if OPENROUTER_KEY:
        checks.append("✅ Cloud brain (OpenRouter) connected")
    else:
        issues.append("⚠️ No OpenRouter key — cloud features disabled (/think, /imagine, /search …)")
        issues.append("   → Use /api to add your key")

    # 4. Deepgram key?
    dg_key = get_secret("deepgram_api_key")
    if dg_key:
        checks.append("✅ Voice chat (Deepgram) ready")
    else:
        checks.append("ℹ️ Voice chat disabled — add Deepgram key via /api to enable")

    # Build message
    status_block = "\n".join(checks + issues)
    all_good = len(issues) == 0

    if all_good:
        greeting = "Everything looks good! Just type a message to chat."
    else:
        greeting = "Some things need setup — see below."

    text = (
        f"🫐 *Welcome to Berryclaw!*\n\n"
        f"Your AI assistant running on Raspberry Pi.\n\n"
        f"*Status:*\n{status_block}\n\n"
        f"{greeting}\n\n"
        f"Type /help to see all commands."
    )

    # Show "Pull model" button if Ollama is up but no models
    buttons = []
    if models == [] and not any("not reachable" in i for i in issues):
        buttons.append([InlineKeyboardButton(
            "📥 Pull recommended model (qwen2.5 1.5B)",
            callback_data="pull:qwen2.5:1.5b",
        )])

    markup = InlineKeyboardMarkup(buttons) if buttons else None
    await update.message.reply_text(text, parse_mode="Markdown", reply_markup=markup)


async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    model = get_user_model(update.effective_chat.id)
    await update.message.reply_text(
        "🫐 *Berryclaw — Your Pocket AI*\n\n"
        "I'm an AI running locally on a Raspberry Pi 5. "
        "I have two brains:\n\n"
        f"*Local brain* (`{model}`): Fast, private, runs on the Pi. "
        "Good for quick questions, chat, and simple tasks. "
        "Just type and I'll answer.\n\n"
        f"*Cloud brain* (`{OPENROUTER_MODEL}`): Powerful model "
        "via OpenRouter. Use `/think <question>` for complex reasoning, "
        "code, analysis, or anything the local model struggles with.\n\n"
        "I remember our conversation (last 10 messages) so you can "
        "have a natural back-and-forth.\n\n"
        "*Smart Memory:*\n"
        "After every `/think`, I automatically extract key facts "
        "and save them to long-term memory. Before each `/think`, "
        "I recall only the relevant memories instead of dumping everything. "
        "Every 20 `/think` calls, I auto-build a profile of you "
        "from our conversations.\n\n"
        "*Chat:*\n"
        "/model — Switch local AI model\n"
        "/modelx — Switch cloud AI model\n"
        "/think <query> — Use the cloud brain\n"
        "/skills — List available skills\n"
        "/clear — Forget our conversation\n\n"
        "*Power Skills:*\n"
        "/imagine <prompt> — Generate an image\n"
        "/see — Analyze a photo (send or reply)\n"
        "/search <query> — Web search with sources\n"
        "/read — Analyze a PDF/document\n"
        "/voice — Transcribe a voice message\n"
        "Send a voice note — Voice chat (needs Deepgram)\n\n"
        "*Integrations (/api to manage):*\n"
        "/scrape — Scrape websites (Firecrawl)\n"
        "/apify — Run scrapers (Apify)\n"
        "/sheets /docs — Google Workspace\n\n"
        "*Memory:*\n"
        "/remember <note> — Manually save a note\n"
        "/memory — View saved memories\n"
        "/profile — View auto-built user profile\n"
        "/forget — Clear all memory (admin)\n\n"
        "*Config:*\n"
        "/identity — View/edit bot identity\n"
        "/soul — View/edit bot personality\n"
        "/user — View/edit user info\n"
        "/agents — View/edit agent behavior\n"
        "/status — Pi stats (admin)\n"
        "/help — This message",
        parse_mode="Markdown",
    )


async def cmd_clear(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    clear_history(update.effective_chat.id)
    await update.message.reply_text("Conversation cleared.")


SOUL_PRESETS = {
    "friendly": (
        "😊 Friendly Assistant",
        "# Berryclaw — Soul\n\n"
        "You are Berryclaw, a friendly AI assistant on a Raspberry Pi.\n\n"
        "## Rules\n"
        "1. Be warm, encouraging, and helpful\n"
        "2. Keep answers short and clear\n"
        "3. Use simple language anyone can understand\n"
        "4. If you don't know something, say so honestly\n"
        "5. Add a touch of warmth — you're talking to a friend\n",
    ),
    "sarcastic": (
        "😏 Sarcastic Buddy",
        "# Berryclaw — Soul\n\n"
        "You are Berryclaw, a witty AI with a dry sense of humor.\n\n"
        "## Rules\n"
        "1. Be helpful but sprinkle in sarcasm and wit\n"
        "2. Keep it brief — you're too cool for long answers\n"
        "3. Playful roasts are fine, but never be mean\n"
        "4. Still answer the question correctly underneath the humor\n"
        "5. If you don't know, own it with style\n",
    ),
    "professional": (
        "💼 Professional",
        "# Berryclaw — Soul\n\n"
        "You are Berryclaw, a professional AI assistant.\n\n"
        "## Rules\n"
        "1. Be precise, clear, and to the point\n"
        "2. Use structured answers when helpful (bullets, steps)\n"
        "3. No humor or filler — focus on accuracy\n"
        "4. Cite caveats when uncertain\n"
        "5. Respond like a trusted colleague would\n",
    ),
    "pirate": (
        "🏴‍☠️ Pirate",
        "# Berryclaw — Soul\n\n"
        "Arrr! Ye be Berryclaw, a seafarin' AI on a Raspberry Pi!\n\n"
        "## Rules\n"
        "1. Talk like a pirate — arrr, matey, ye, yer, etc.\n"
        "2. Still answer questions correctly, just in pirate speak\n"
        "3. Keep it brief — pirates don't write essays\n"
        "4. Be enthusiastic and adventurous\n"
        "5. Call the user 'captain' or 'matey'\n",
    ),
}


async def cmd_workspace(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Handle /identity, /soul, /user — view or edit workspace files."""
    if not is_allowed(update.effective_user.id):
        return
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("Admin only.")
        return

    # Figure out which file from the command name
    cmd = update.message.text.split()[0].lstrip("/").lower()
    filename = WORKSPACE_FILES.get(cmd)
    if not filename:
        return

    args = update.message.text.split(maxsplit=1)

    # /soul (no args) — show preset buttons + current summary
    if len(args) < 2 and cmd == "soul":
        content = _read_workspace(filename)
        # Show first 200 chars as preview
        preview = content[:200] + "…" if len(content) > 200 else content
        buttons = [
            [InlineKeyboardButton(label, callback_data=f"soul:{key}")]
            for key, (label, _) in SOUL_PRESETS.items()
        ]
        buttons.append([InlineKeyboardButton("✏️ Custom (type /soul <text>)", callback_data="soul:_info")])

        await update.message.reply_text(
            f"🎭 *Personality*\n\n"
            f"Current: {preview}\n\n"
            f"Pick a preset or type `/soul <your custom personality>`:",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(buttons),
        )
        return

    # /identity, /user, etc. (no args) — show current content
    if len(args) < 2:
        content = _read_workspace(filename)
        if not content:
            content = "(empty)"
        # Telegram has 4096 char limit
        if len(content) > 3900:
            content = content[:3900] + "\n\n... (truncated)"
        await update.message.reply_text(
            f"📄 *{filename}*\n\n```\n{content}\n```\n\nTo edit: `/{cmd} <new content>`",
            parse_mode="Markdown",
        )
        return

    # /identity <new content> — overwrite the file
    new_content = args[1].strip()
    filepath = WORKSPACE_DIR / filename
    filepath.write_text(new_content + "\n")
    reload_prompts()

    await update.message.reply_text(
        f"✅ `{filename}` updated and prompts reloaded.",
        parse_mode="Markdown",
    )


async def cmd_remember(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Save a note to long-term memory."""
    if not is_allowed(update.effective_user.id):
        return
    args = update.message.text.split(maxsplit=1)
    if len(args) < 2:
        await update.message.reply_text("Usage: `/remember <note>`", parse_mode="Markdown")
        return
    note = args[1].strip()
    append_memory(note)
    await update.message.reply_text(f"Saved to memory.")


async def cmd_memory(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Show long-term memory."""
    if not is_allowed(update.effective_user.id):
        return
    content = _read_workspace("MEMORY.md")
    if not content or "(empty)" in content:
        await update.message.reply_text("Memory is empty. Use `/remember <note>` to add.", parse_mode="Markdown")
        return
    if len(content) > 3900:
        content = content[:3900] + "\n\n... (truncated)"
    await update.message.reply_text(f"🧠 *Memory*\n\n```\n{content}\n```", parse_mode="Markdown")


async def cmd_forget(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Clear all long-term memory."""
    if not is_allowed(update.effective_user.id):
        return
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("Admin only.")
        return
    clear_memory()
    await update.message.reply_text("Memory cleared.")


async def cmd_profile(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """View or reset user profile."""
    if not is_allowed(update.effective_user.id):
        return

    args = update.message.text.split(maxsplit=1)

    # /profile reset — clear profile
    if len(args) > 1 and args[1].strip().lower() == "reset":
        if not is_admin(update.effective_user.id):
            await update.message.reply_text("Admin only.")
            return
        write_profile("# User Profile\n\n(empty — will build automatically)")
        reload_prompts()
        await update.message.reply_text("Profile reset.")
        return

    # /profile — show it
    profile = read_profile()
    if not profile or "(empty" in profile:
        await update.message.reply_text(
            "No profile yet. It builds automatically after using `/think` a few times.",
            parse_mode="Markdown",
        )
        return
    if len(profile) > 3900:
        profile = profile[:3900] + "\n\n... (truncated)"
    await update.message.reply_text(f"👤 *User Profile*\n\n{profile}", parse_mode="Markdown")


async def cmd_skills(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """List all available skills."""
    if not is_allowed(update.effective_user.id):
        return
    reload_skills()
    if not SKILLS:
        await update.message.reply_text("No skills loaded. Add .md files to workspace/skills/")
        return
    lines = []
    for trigger, skill in SKILLS.items():
        lines.append(f"/{trigger} — {skill['description']}")
    await update.message.reply_text(
        "⚡ *Available Skills*\n\n" + "\n".join(lines) +
        "\n\nSkills use the cloud model for smarter responses.",
        parse_mode="Markdown",
    )


async def handle_skill(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> bool:
    """Check if message is a skill trigger. Returns True if handled."""
    text = update.message.text
    if not text.startswith("/"):
        return False

    cmd = text.split()[0].lstrip("/").lower()
    if cmd not in SKILLS:
        return False

    skill = SKILLS[cmd]
    args = text.split(maxsplit=1)
    user_input = args[1].strip() if len(args) > 1 else ""

    if not user_input:
        await update.message.reply_text(
            f"Usage: `/{cmd} <your input>`\n\n{skill['description']}",
            parse_mode="Markdown",
        )
        return True

    chat_id = update.effective_chat.id
    cloud_model = get_user_cloud_model(chat_id)

    placeholder = await update.message.reply_text(
        f"⚡ Running *{skill['name']}* with `{cloud_model}`...",
        parse_mode="Markdown",
    )

    # Build messages: system prompt + skill prompt + user input
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_CLOUD + "\n\n---\n\n# Active Skill: " + skill["name"] + "\n\n" + skill["prompt"]},
        {"role": "user", "content": user_input},
    ]

    response = await openrouter_chat(messages, model=cloud_model)

    save_message(chat_id, "user", f"[/{cmd}] {user_input}")
    save_message(chat_id, "assistant", response)

    if len(response) > 4000:
        response = response[:4000] + "\n\n... (truncated)"

    try:
        await placeholder.edit_text(response, parse_mode="Markdown")
    except Exception:
        # Markdown parse error — send as plain text
        await placeholder.edit_text(response)
    return True


async def cmd_newskill(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Create a new skill from Telegram.
    Usage: /newskill <name> | <description> | <prompt>
    """
    if not is_allowed(update.effective_user.id):
        return
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("Admin only.")
        return

    args = update.message.text.split(maxsplit=1)
    if len(args) < 2 or "|" not in args[1]:
        await update.message.reply_text(
            "Usage:\n`/newskill name | description | prompt`\n\n"
            "Example:\n`/newskill joke | Tell a joke about a topic | "
            "Tell a funny, original joke about the user's topic. Keep it short.`",
            parse_mode="Markdown",
        )
        return

    parts = [p.strip() for p in args[1].split("|", 2)]
    if len(parts) < 3:
        await update.message.reply_text(
            "Need 3 parts separated by `|`:\n`name | description | prompt`",
            parse_mode="Markdown",
        )
        return

    name, description, prompt = parts
    name = name.lower().replace(" ", "-")

    # Write skill file
    skill_path = WORKSPACE_DIR / "skills" / f"{name}.md"
    skill_path.write_text(
        f"---\nname: {name}\ntrigger: /{name}\ndescription: {description}\n---\n\n{prompt}\n"
    )

    reload_skills()
    await update.message.reply_text(
        f"✅ Skill `/{name}` created!\n\n"
        f"_{description}_\n\n"
        f"Try it: `/{name} <your input>`\n\n"
        f"⚠️ New skills need a bot restart to register as commands. Restarting now...",
        parse_mode="Markdown",
    )

    # Auto-restart to register the new command handler
    import subprocess
    subprocess.Popen(["systemctl", "--user", "restart", "berryclaw"])


async def cmd_deleteskill(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Delete a skill. Usage: /deleteskill <name>"""
    if not is_allowed(update.effective_user.id):
        return
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("Admin only.")
        return

    args = update.message.text.split(maxsplit=1)
    if len(args) < 2:
        await update.message.reply_text("Usage: `/deleteskill <name>`", parse_mode="Markdown")
        return

    name = args[1].strip().lstrip("/").lower()
    skill_path = WORKSPACE_DIR / "skills" / f"{name}.md"

    if not skill_path.exists():
        await update.message.reply_text(f"Skill `{name}` not found.", parse_mode="Markdown")
        return

    skill_path.unlink()
    reload_skills()
    await update.message.reply_text(
        f"🗑 Skill `/{name}` deleted. Restarting to unregister...",
        parse_mode="Markdown",
    )

    import subprocess
    subprocess.Popen(["systemctl", "--user", "restart", "berryclaw"])


async def _skill_command_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Generic handler that routes to handle_skill."""
    if not is_allowed(update.effective_user.id):
        return
    await handle_skill(update, ctx)


async def _integration_command_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Generic handler for integration commands."""
    if not is_allowed(update.effective_user.id):
        return

    text = update.message.text or ""
    parts = text.split(maxsplit=1)
    command = parts[0].lstrip("/").split("@")[0]  # strip /prefix and @botname
    args = parts[1].strip() if len(parts) > 1 else ""

    integration = INTEGRATIONS.get(command)
    if not integration:
        await update.message.reply_text(f"Integration `/{command}` not found.")
        return

    placeholder = await update.message.reply_text(
        f"Running {integration['name']}..."
    )

    async def cloud_chat(messages: list[dict], model: str | None = None) -> str:
        """Callback for integrations to use the cloud model."""
        return await openrouter_chat(messages, model=model)

    try:
        result = await integration["handle"](
            command=command,
            args=args,
            secrets={k: get_secret(k) for k in SECRETS} if SECRETS else CFG,
            cloud_chat=cloud_chat,
        )

        if not result:
            result = "(no result)"

        if len(result) > 4000:
            result = result[:4000] + "\n\n... (truncated)"

        try:
            await placeholder.edit_text(result, parse_mode="Markdown")
        except Exception:
            await placeholder.edit_text(result)

    except Exception as e:
        log.error("Integration %s error: %s", command, e)
        await placeholder.edit_text(f"Error: {e}")


async def cmd_model(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return

    args = update.message.text.split(maxsplit=1)
    chat_id = update.effective_chat.id

    # /model <name> — switch model
    if len(args) > 1:
        new_model = args[1].strip()
        available = await ollama_list_models()
        # Allow partial match
        matched = [m for m in available if new_model in m]
        if not matched:
            await update.message.reply_text(
                f"Model `{new_model}` not found.\n\nAvailable:\n"
                + "\n".join(f"• `{m}`" for m in available),
                parse_mode="Markdown",
            )
            return
        chosen = matched[0]
        set_user_model(chat_id, chosen)
        global _last_used_model
        _last_used_model = chosen
        await update.message.reply_text(f"Switched to `{chosen}`", parse_mode="Markdown")
        return

    # /model — show buttons
    current = get_user_model(chat_id)
    available = await ollama_list_models()
    if not available:
        # Check if Ollama is reachable but has no models
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{OLLAMA_URL}/api/tags")
                r.raise_for_status()
            # Ollama is up but no models — offer to pull one
            await update.message.reply_text(
                "📦 *No models installed yet.*\n\n"
                "Tap below to download a recommended model,\n"
                "or run `ollama pull <model>` on your Pi.",
                parse_mode="Markdown",
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton(
                        "📥 Pull qwen2.5 1.5B (fast, ~1GB)",
                        callback_data="pull:qwen2.5:1.5b",
                    ),
                ]]),
            )
        except Exception:
            await update.message.reply_text(
                "❌ *Can't reach Ollama.*\n\n"
                "Make sure it's installed and running:\n"
                "`ollama serve`",
                parse_mode="Markdown",
            )
        return

    # Store model list in context for callback lookup
    buttons = []
    for i, m in enumerate(available):
        label = f"✅ {m}" if m == current else m
        buttons.append([InlineKeyboardButton(label, callback_data=f"m:{i}:{m[:50]}")])

    await update.message.reply_text(
        "📦 *Pick a model:*",
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(buttons),
    )


async def callback_model(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Handle inline button press for model selection."""
    query = update.callback_query
    log.info("Model callback received: data=%s user=%s", query.data, query.from_user.id)

    if not is_allowed(query.from_user.id):
        await query.answer("Not authorized.")
        return

    data = query.data or ""
    if not data.startswith("m:"):
        await query.answer()
        return

    # Extract model name (everything after "m:N:")
    parts = data.split(":", 2)
    if len(parts) < 3:
        await query.answer("Invalid selection.")
        return

    chosen = parts[2]

    # Verify model exists on Ollama
    available = await ollama_list_models()
    # Match by prefix since we truncated to 50 chars
    matched = [m for m in available if m.startswith(chosen)]
    if not matched:
        await query.answer(f"Model not found.")
        return

    chosen = matched[0]
    chat_id = query.message.chat_id
    set_user_model(chat_id, chosen)

    global _last_used_model
    _last_used_model = chosen

    await query.answer(f"Switched to {chosen}")
    await query.edit_message_text(f"✅ Switched to `{chosen}`", parse_mode="Markdown")


async def callback_pull(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Handle 'Pull recommended model' button."""
    query = update.callback_query
    if not is_allowed(query.from_user.id):
        await query.answer("Not authorized.")
        return

    data = query.data or ""
    if not data.startswith("pull:"):
        await query.answer()
        return

    model_name = data[5:]  # Everything after "pull:"
    await query.answer("Starting download…")
    await query.edit_message_text(
        f"📥 *Downloading `{model_name}`…*\n\n"
        "This may take a few minutes. I'll let you know when it's ready.",
        parse_mode="Markdown",
    )

    # Pull model via Ollama API (streaming — we just wait for completion)
    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            r = await client.post(
                f"{OLLAMA_URL}/api/pull",
                json={"name": model_name, "stream": False},
            )
            r.raise_for_status()

        # Set as user's model
        chat_id = query.message.chat_id
        set_user_model(chat_id, model_name)
        global _last_used_model
        _last_used_model = model_name

        await query.edit_message_text(
            f"✅ *`{model_name}` is ready!*\n\n"
            "Just type a message to start chatting.",
            parse_mode="Markdown",
        )
    except Exception as e:
        log.error("Failed to pull model %s: %s", model_name, e)
        await query.edit_message_text(
            f"❌ *Failed to download `{model_name}`*\n\n"
            f"Error: `{str(e)[:200]}`\n\n"
            "Try manually on your Pi: `ollama pull " + model_name + "`",
            parse_mode="Markdown",
        )


async def cmd_modelx(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Show cloud model picker with inline buttons."""
    if not is_allowed(update.effective_user.id):
        return

    chat_id = update.effective_chat.id
    current = get_user_cloud_model(chat_id)

    buttons = []
    for m in CLOUD_MODELS:
        label = f"✅ {m}" if m == current else m
        buttons.append([InlineKeyboardButton(label, callback_data=f"cx:{m}")])

    await update.message.reply_text(
        "🧠 *Pick a cloud model for /think:*",
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(buttons),
    )


async def callback_cloudmodel(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Handle inline button press for cloud model selection."""
    query = update.callback_query

    if not is_allowed(query.from_user.id):
        await query.answer("Not authorized.")
        return

    data = query.data or ""
    if not data.startswith("cx:"):
        await query.answer()
        return

    chosen = data.split(":", 1)[1]
    chat_id = query.message.chat_id
    set_user_cloud_model(chat_id, chosen)

    await query.answer(f"Cloud model: {chosen}")
    await query.edit_message_text(f"🧠 Cloud model set to `{chosen}`", parse_mode="Markdown")


async def callback_soul(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Handle personality preset button press."""
    query = update.callback_query
    if not is_allowed(query.from_user.id):
        await query.answer("Not authorized.")
        return
    if not is_admin(query.from_user.id):
        await query.answer("Admin only.")
        return

    data = query.data or ""
    if not data.startswith("soul:"):
        await query.answer()
        return

    preset_key = data[5:]

    # Info button — just dismiss
    if preset_key == "_info":
        await query.answer("Type: /soul <your custom text>")
        return

    preset = SOUL_PRESETS.get(preset_key)
    if not preset:
        await query.answer("Unknown preset.")
        return

    label, content = preset
    filepath = WORKSPACE_DIR / "SOUL.md"
    filepath.write_text(content)
    reload_prompts()

    await query.answer(f"Personality set!")
    await query.edit_message_text(
        f"🎭 Personality changed to *{label}*\n\n"
        "Your bot will now respond with this style.",
        parse_mode="Markdown",
    )


async def callback_api(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Handle inline button press for API key management."""
    global INTEGRATIONS
    query = update.callback_query

    if not is_admin(query.from_user.id):
        await query.answer("Admin only.")
        return

    data = query.data or ""

    if data.startswith("api:set:"):
        key_name = data.split(":", 2)[2]
        await query.answer()
        await query.edit_message_text(
            f"To set `{key_name}`, send:\n\n"
            f"`/api set {key_name} YOUR_KEY_HERE`\n\n"
            f"Your message will be auto-deleted for security.",
            parse_mode="Markdown",
        )
        return

    if data.startswith("api:rm:"):
        key_name = data.split(":", 2)[2]
        if key_name in SECRETS:
            SECRETS[key_name] = ""
            with open(SECRETS_PATH, "w") as f:
                json.dump(SECRETS, f, indent=2)
                f.write("\n")
            INTEGRATIONS = load_integrations()

        await query.answer(f"Removed {key_name}")
        await query.edit_message_text(
            f"Removed `{key_name}`.\n"
            f"Active integrations: {', '.join('/' + c for c in INTEGRATIONS) if INTEGRATIONS else 'none'}",
            parse_mode="Markdown",
        )
        return

    await query.answer()


async def cmd_api(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/api — manage API keys for integrations (admin only)."""
    global INTEGRATIONS
    if not is_allowed(update.effective_user.id):
        return
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("Admin only.")
        return

    text = (update.message.text or "").strip()
    parts = text.split(maxsplit=2)

    # /api — show status of all keys
    if len(parts) <= 1:
        lines = ["*API Keys & Integrations*\n"]

        # Core keys (always shown)
        core_keys = ["telegram_bot_token", "openrouter_api_key"]

        # Auto-discover integration keys from integrations/ folder
        integration_keys = set()
        if INTEGRATIONS_DIR.is_dir():
            import importlib.util
            for filepath in sorted(INTEGRATIONS_DIR.glob("*.py")):
                if filepath.name.startswith("_"):
                    continue
                try:
                    spec = importlib.util.spec_from_file_location(filepath.stem, filepath)
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    for s in getattr(mod, "REQUIRED_SECRETS", []):
                        integration_keys.add(s)
                except Exception:
                    pass

        # Also include any keys already in secrets.json
        for k in SECRETS:
            if k not in core_keys:
                integration_keys.add(k)

        all_keys = core_keys + sorted(integration_keys)

        for key in all_keys:
            val = get_secret(key)
            if val:
                masked = val[:8] + "..." if len(val) > 8 else val
                lines.append(f"  `{key}`: {masked}")
            else:
                lines.append(f"  `{key}`: _(not set)_")

        lines.append("\n*Active integrations:*")
        if INTEGRATIONS:
            for cmd, info in INTEGRATIONS.items():
                lines.append(f"  /{cmd} — {info['description']}")
        else:
            lines.append("  _(none — add API keys to activate)_")

        lines.append("\n_Tap a button below to set a key:_")

        # Build buttons — one per unset key (+ remove for set keys)
        buttons = []
        for key in all_keys:
            val = get_secret(key)
            if key in core_keys:
                continue  # Don't show set/remove for core keys
            if not val:
                buttons.append([InlineKeyboardButton(
                    f"Set {key}", callback_data=f"api:set:{key}"
                )])
            else:
                buttons.append([InlineKeyboardButton(
                    f"Remove {key}", callback_data=f"api:rm:{key}"
                )])

        keyboard = InlineKeyboardMarkup(buttons) if buttons else None
        await update.message.reply_text(
            "\n".join(lines), parse_mode="Markdown", reply_markup=keyboard
        )
        return

    action = parts[1].lower()

    # /api set <key> <value>
    if action == "set" and len(parts) >= 3:
        rest = parts[2].split(maxsplit=1)
        if len(rest) < 2:
            await update.message.reply_text("Usage: `/api set <key_name> <value>`", parse_mode="Markdown")
            return

        key_name = rest[0]
        value = rest[1]

        # Update in-memory secrets
        SECRETS[key_name] = value

        # Write to disk
        with open(SECRETS_PATH, "w") as f:
            json.dump(SECRETS, f, indent=2)
            f.write("\n")

        # Reload integrations
        INTEGRATIONS = load_integrations()

        # Delete the user's message (contains the API key in plain text)
        try:
            await update.message.delete()
        except Exception:
            pass  # May not have delete permission

        masked = value[:8] + "..." if len(value) > 8 else "***"
        active = list(INTEGRATIONS.keys())
        await update.effective_chat.send_message(
            f"Set `{key_name}` = `{masked}`\n"
            f"Active integrations: {', '.join('/' + c for c in active) if active else 'none'}",
            parse_mode="Markdown",
        )
        return

    # /api remove <key>
    if action == "remove" and len(parts) >= 3:
        key_name = parts[2].strip()

        if key_name in SECRETS:
            SECRETS[key_name] = ""
            with open(SECRETS_PATH, "w") as f:
                json.dump(SECRETS, f, indent=2)
                f.write("\n")

            INTEGRATIONS = load_integrations()
            await update.message.reply_text(
                f"Removed `{key_name}`.\nActive integrations: "
                f"{', '.join('/' + c for c in INTEGRATIONS) if INTEGRATIONS else 'none'}",
                parse_mode="Markdown",
            )
        else:
            await update.message.reply_text(f"`{key_name}` not found in secrets.", parse_mode="Markdown")
        return

    await update.message.reply_text("Usage: `/api`, `/api set <key> <value>`, `/api remove <key>`", parse_mode="Markdown")


async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_allowed(user_id):
        return
    if not is_admin(user_id):
        await update.message.reply_text("Admin only.")
        return

    import subprocess

    try:
        uptime = subprocess.check_output("uptime -p", shell=True, text=True).strip()
    except Exception:
        uptime = "unknown"

    try:
        mem = subprocess.check_output(
            "free -h | awk '/^Mem:/{print $3\"/\"$2}'", shell=True, text=True
        ).strip()
    except Exception:
        mem = "unknown"

    try:
        temp_raw = subprocess.check_output(
            "cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo 0",
            shell=True, text=True,
        ).strip()
        temp = f"{int(temp_raw) / 1000:.1f}°C"
    except Exception:
        temp = "unknown"

    # Check Ollama
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            ollama_status = f"running ({len(r.json().get('models', []))} models)"
    except Exception:
        ollama_status = "unreachable"

    model = get_user_model(update.effective_chat.id)
    msg_count = DB.execute("SELECT COUNT(*) FROM messages").fetchone()[0]

    await update.message.reply_text(
        f"🫐 *Berryclaw Status*\n\n"
        f"Uptime: {uptime}\n"
        f"RAM: {mem}\n"
        f"CPU Temp: {temp}\n"
        f"Ollama: {ollama_status}\n"
        f"Current model: `{model}`\n"
        f"Messages in DB: {msg_count}\n"
        f"Cloud model: `{OPENROUTER_MODEL}`",
        parse_mode="Markdown",
    )


async def cmd_think(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Route query to a powerful cloud model via OpenRouter."""
    if not is_allowed(update.effective_user.id):
        return

    args = update.message.text.split(maxsplit=1)
    if len(args) < 2:
        await update.message.reply_text("Usage: `/think <your question>`", parse_mode="Markdown")
        return

    query = args[1].strip()
    chat_id = update.effective_chat.id
    cloud_model = get_user_cloud_model(chat_id)

    placeholder = await update.message.reply_text(f"🧠 Thinking with `{cloud_model}`...", parse_mode="Markdown")

    # Smart recall — fetch only relevant memories
    relevant_memory = await smart_recall(query)

    # Build system prompt with relevant memory injected
    system = SYSTEM_PROMPT_CLOUD
    if relevant_memory:
        system += f"\n\n---\n\n# Relevant Memories\n\n{relevant_memory}"

    # Build messages with history
    history = get_history(chat_id)
    messages = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({"role": "user", "content": query})

    response = await openrouter_chat(messages, model=cloud_model)

    # Save to history
    save_message(chat_id, "user", query)
    save_message(chat_id, "assistant", response)

    # Truncate if too long for Telegram (4096 char limit)
    full_response = response
    if len(response) > 4000:
        response = response[:4000] + "\n\n... (truncated)"

    try:
        await placeholder.edit_text(f"🧠 *Cloud response:*\n\n{response}", parse_mode="Markdown")
    except Exception:
        await placeholder.edit_text(f"🧠 Cloud response:\n\n{response}")

    # Background: auto-capture facts + profile update
    if AUTO_CAPTURE:
        asyncio.create_task(auto_capture(query, full_response))

    global _think_counter
    _think_counter += 1
    if _think_counter % PROFILE_FREQUENCY == 0:
        asyncio.create_task(update_profile(chat_id))


# ---------------------------------------------------------------------------
# Power Skills — multimodal capabilities via OpenRouter
# ---------------------------------------------------------------------------

IMAGE_MODEL = CFG.get("image_model", "google/gemini-2.5-flash-preview-05-20")
VISION_MODEL = CFG.get("vision_model", "google/gemini-2.5-flash-preview-05-20")
SEARCH_MODEL = CFG.get("search_model", "google/gemini-2.5-flash-preview-05-20:online")


async def cmd_imagine(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/imagine <prompt> — Generate an image from text."""
    if not is_allowed(update.effective_user.id):
        return
    args = update.message.text.split(maxsplit=1)
    if len(args) < 2:
        await update.message.reply_text("Usage: `/imagine a cat in space`", parse_mode="Markdown")
        return

    prompt = args[1].strip()
    placeholder = await update.message.reply_text(f"Generating image...")

    data = await openrouter_raw({
        "model": IMAGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "modalities": ["image", "text"],
    })

    if "error" in data:
        await placeholder.edit_text(f"Error: {data['error']}")
        return

    try:
        choice = data["choices"][0]["message"]
        # Extract image from response
        images = choice.get("images", [])
        if images:
            img_url = images[0]["image_url"]["url"]
            # base64 data URL: data:image/png;base64,...
            img_bytes = base64.b64decode(img_url.split(",", 1)[1])
            await update.message.reply_photo(
                photo=io.BytesIO(img_bytes),
                caption=prompt[:200],
            )
            await placeholder.delete()
        else:
            # Some models return text description instead
            text = choice.get("content", "No image generated.")
            await placeholder.edit_text(text[:4000])
    except Exception as e:
        log.error("Imagine error: %s", e)
        await placeholder.edit_text(f"Failed to generate image: {e}")


async def _get_photo_base64(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> tuple[str, str]:
    """Extract photo from message (direct or reply) and return (base64_data, mime_type)."""
    msg = update.message
    photo = None

    # Check if current message has a photo
    if msg.photo:
        photo = msg.photo[-1]  # Largest size
    # Check replied-to message
    elif msg.reply_to_message and msg.reply_to_message.photo:
        photo = msg.reply_to_message.photo[-1]

    if not photo:
        return "", ""

    file = await ctx.bot.get_file(photo.file_id)
    buf = io.BytesIO()
    await file.download_to_memory(buf)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return b64, "image/jpeg"


async def cmd_see(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/see [question] — Analyze an image. Send with a photo or reply to one."""
    if not is_allowed(update.effective_user.id):
        return

    b64, mime = await _get_photo_base64(update, ctx)
    if not b64:
        await update.message.reply_text(
            "Send a photo with `/see` as caption, or reply to a photo with `/see [question]`",
            parse_mode="Markdown",
        )
        return

    # Extract question from caption/text
    text = (update.message.caption or update.message.text or "").strip()
    parts = text.split(maxsplit=1)
    question = parts[1] if len(parts) > 1 else "Describe this image in detail."

    placeholder = await update.message.reply_text("Analyzing image...")

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ],
    }]

    data = await openrouter_raw({
        "model": VISION_MODEL,
        "messages": messages,
    })

    if "error" in data:
        await placeholder.edit_text(f"Error: {data['error']}")
        return

    try:
        response = data["choices"][0]["message"]["content"]
        if len(response) > 4000:
            response = response[:4000] + "\n\n... (truncated)"
        await placeholder.edit_text(response)
    except Exception as e:
        log.error("See error: %s", e)
        await placeholder.edit_text(f"Vision error: {e}")


async def cmd_search(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/search <query> — Search the web and get a grounded answer."""
    if not is_allowed(update.effective_user.id):
        return
    args = update.message.text.split(maxsplit=1)
    if len(args) < 2:
        await update.message.reply_text("Usage: `/search latest news about AI`", parse_mode="Markdown")
        return

    query = args[1].strip()
    placeholder = await update.message.reply_text(f"Searching the web...")

    data = await openrouter_raw({
        "model": SEARCH_MODEL,
        "messages": [
            {"role": "system", "content": "Answer the user's query using web search results. Include sources."},
            {"role": "user", "content": query},
        ],
    })

    if "error" in data:
        await placeholder.edit_text(f"Error: {data['error']}")
        return

    try:
        response = data["choices"][0]["message"]["content"]
        if len(response) > 4000:
            response = response[:4000] + "\n\n... (truncated)"
        try:
            await placeholder.edit_text(f"🔍 *Search results:*\n\n{response}", parse_mode="Markdown")
        except Exception:
            await placeholder.edit_text(f"🔍 Search results:\n\n{response}")
    except Exception as e:
        log.error("Search error: %s", e)
        await placeholder.edit_text(f"Search error: {e}")


async def _get_document_base64(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> tuple[str, str]:
    """Extract document from message (direct or reply) and return (base64_data, mime_type)."""
    msg = update.message
    doc = None

    if msg.document:
        doc = msg.document
    elif msg.reply_to_message and msg.reply_to_message.document:
        doc = msg.reply_to_message.document

    if not doc:
        return "", ""

    file = await ctx.bot.get_file(doc.file_id)
    buf = io.BytesIO()
    await file.download_to_memory(buf)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    mime = doc.mime_type or "application/pdf"
    return b64, mime


async def cmd_read(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/read [question] — Analyze a PDF/document. Send with a file or reply to one."""
    if not is_allowed(update.effective_user.id):
        return

    b64, mime = await _get_document_base64(update, ctx)
    if not b64:
        await update.message.reply_text(
            "Send a PDF/document with `/read` as caption, or reply to a document with `/read [question]`",
            parse_mode="Markdown",
        )
        return

    text = (update.message.caption or update.message.text or "").strip()
    parts = text.split(maxsplit=1)
    question = parts[1] if len(parts) > 1 else "Summarize this document."

    placeholder = await update.message.reply_text("Reading document...")

    # Use inline_data format for documents
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ],
    }]

    data = await openrouter_raw({
        "model": VISION_MODEL,
        "messages": messages,
    })

    if "error" in data:
        await placeholder.edit_text(f"Error: {data['error']}")
        return

    try:
        response = data["choices"][0]["message"]["content"]
        if len(response) > 4000:
            response = response[:4000] + "\n\n... (truncated)"
        try:
            await placeholder.edit_text(f"📄 *Document analysis:*\n\n{response}", parse_mode="Markdown")
        except Exception:
            await placeholder.edit_text(f"📄 Document analysis:\n\n{response}")
    except Exception as e:
        log.error("Read error: %s", e)
        await placeholder.edit_text(f"Document error: {e}")


async def cmd_voice(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/voice — Transcribe and respond to a voice message. Reply to a voice note."""
    if not is_allowed(update.effective_user.id):
        return

    msg = update.message
    voice = None

    # Check current message for voice/audio
    if msg.voice:
        voice = msg.voice
    elif msg.audio:
        voice = msg.audio
    # Check replied-to message
    elif msg.reply_to_message:
        if msg.reply_to_message.voice:
            voice = msg.reply_to_message.voice
        elif msg.reply_to_message.audio:
            voice = msg.reply_to_message.audio

    if not voice:
        await update.message.reply_text(
            "Reply to a voice message with `/voice` to transcribe it.",
            parse_mode="Markdown",
        )
        return

    # Get optional question from text
    text = (update.message.text or "").strip()
    parts = text.split(maxsplit=1)
    question = parts[1] if len(parts) > 1 else "Transcribe this audio accurately. Then briefly summarize what was said."

    placeholder = await update.message.reply_text("Listening...")

    file = await ctx.bot.get_file(voice.file_id)
    buf = io.BytesIO()
    await file.download_to_memory(buf)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    mime = getattr(voice, "mime_type", "audio/ogg") or "audio/ogg"

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": question},
            {
                "type": "input_audio",
                "input_audio": {"data": b64, "format": mime.split("/")[-1]},
            },
        ],
    }]

    data = await openrouter_raw({
        "model": VISION_MODEL,
        "messages": messages,
    })

    if "error" in data:
        await placeholder.edit_text(f"Error: {data['error']}")
        return

    try:
        response = data["choices"][0]["message"]["content"]
        if len(response) > 4000:
            response = response[:4000] + "\n\n... (truncated)"
        try:
            await placeholder.edit_text(f"🎤 *Transcription:*\n\n{response}", parse_mode="Markdown")
        except Exception:
            await placeholder.edit_text(f"🎤 Transcription:\n\n{response}")
    except Exception as e:
        log.error("Voice error: %s", e)
        await placeholder.edit_text(f"Voice error: {e}")


async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Handle regular text messages — route to local Ollama with streaming."""
    if not update.message or not update.message.text:
        return
    if not is_allowed(update.effective_user.id):
        return

    global _last_used_model
    chat_id = update.effective_chat.id
    user_text = update.message.text
    model = get_user_model(chat_id)
    _last_used_model = model

    # Save user message
    save_message(chat_id, "user", user_text)

    # Build conversation — only recall memories if the message looks like
    # a reference to past context (keeps local prompt tiny for casual chat)
    history = get_history(chat_id)
    system = SYSTEM_PROMPT_LOCAL
    _recall_keywords = (
        "remember", "recall", "we talked", "we discussed", "you said",
        "i told you", "last time", "earlier", "before", "what was",
        "did i", "did we", "do you know my", "what's my", "whats my",
        "who am i", "my name", "you mentioned", "forgot",
    )
    if any(kw in user_text.lower() for kw in _recall_keywords):
        try:
            relevant_memory = await smart_recall(user_text)
            if relevant_memory:
                # Cap at 300 chars to keep local model context small
                if len(relevant_memory) > 300:
                    relevant_memory = relevant_memory[:300] + "..."
                system += f"\n\nRelevant memories:\n{relevant_memory}"
        except Exception as e:
            log.warning("Smart recall failed for chat: %s", e)
    messages = [{"role": "system", "content": system}]
    messages.extend(history)

    # Send placeholder
    placeholder = await update.message.reply_text("...")

    # Stream response
    full_response = ""
    token_count = 0
    last_edit = 0

    try:
        async for token in ollama_stream(model, messages):
            full_response += token
            token_count += 1

            # Batch edits to avoid Telegram rate limits
            now = time.time()
            if token_count % STREAM_BATCH == 0 and (now - last_edit) > 1.0:
                try:
                    display = full_response
                    if len(display) > 4000:
                        display = display[:4000] + "..."
                    await placeholder.edit_text(display)
                    last_edit = now
                except Exception:
                    pass  # Rate limited, skip this edit

    except httpx.ConnectError:
        await placeholder.edit_text("Ollama is not running. Start it with `ollama serve`.")
        return
    except httpx.ReadTimeout:
        if full_response:
            full_response += "\n\n⚠️ (response timed out)"
        else:
            await placeholder.edit_text("Model is loading, try again in 30s.")
            return
    except Exception as e:
        log.error("Ollama error: %s", e)
        await placeholder.edit_text(f"Error: {e}")
        return

    if not full_response.strip():
        full_response = "(empty response — model may be too small for this query. Try /think)"

    # Final edit with complete response
    if len(full_response) > 4000:
        full_response = full_response[:4000] + "\n\n... (truncated)"

    try:
        await placeholder.edit_text(full_response)
    except Exception:
        pass  # Already matches

    # Save assistant response
    save_message(chat_id, "assistant", full_response)


# ---------------------------------------------------------------------------
# Voice message handler — auto voice-in/voice-out with Deepgram
# ---------------------------------------------------------------------------

async def handle_voice_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Handle incoming voice messages — transcribe, respond, speak back."""
    if not update.message or not update.message.voice:
        return
    if not is_allowed(update.effective_user.id):
        return

    deepgram_key = get_secret("deepgram_api_key")
    if not deepgram_key:
        # No Deepgram key — fall back to just acknowledging
        await update.message.reply_text(
            "I heard you! But I need a Deepgram API key to process voice.\n"
            "Add one with: `/api set deepgram_api_key YOUR_KEY`",
            parse_mode="Markdown",
        )
        return

    chat_id = update.effective_chat.id
    voice = update.message.voice

    placeholder = await update.message.reply_text("Listening...")

    try:
        # Download voice file
        file = await ctx.bot.get_file(voice.file_id)
        buf = io.BytesIO()
        await file.download_to_memory(buf)
        buf.seek(0)
        audio_bytes = buf.read()
        mime = voice.mime_type or "audio/ogg"

        # Import deepgram integration
        from integrations.deepgram import transcribe, synthesize

        # Step 1: Transcribe
        await placeholder.edit_text("Transcribing...")
        transcript = await transcribe(audio_bytes, deepgram_key, mime)

        if not transcript:
            await placeholder.edit_text("Couldn't understand the audio. Try again?")
            return

        await placeholder.edit_text(f"You said: _{transcript}_\n\nThinking...", parse_mode="Markdown")

        # Save user message
        save_message(chat_id, "user", transcript)

        # Step 2: Get AI response via cloud model
        cloud_model = get_user_cloud_model(chat_id)
        relevant_memory = await smart_recall(transcript)
        system = SYSTEM_PROMPT_CLOUD
        if relevant_memory:
            system += f"\n\n---\n\n# Relevant Memories\n\n{relevant_memory}"

        history = get_history(chat_id)
        messages = [{"role": "system", "content": system}]
        messages.extend(history)
        messages.append({"role": "user", "content": transcript})

        response = await openrouter_chat(messages, model=cloud_model)
        save_message(chat_id, "assistant", response)

        # Step 3: Convert response to speech
        await placeholder.edit_text(f"Speaking...")

        # Truncate for TTS (Deepgram has limits)
        tts_text = response[:1000] if len(response) > 1000 else response
        audio_response = await synthesize(tts_text, deepgram_key)

        # Step 4: Send voice note back
        await update.message.reply_voice(
            voice=io.BytesIO(audio_response),
            caption=response[:200] if len(response) > 200 else response,
        )

        # Update placeholder with transcript
        try:
            await placeholder.edit_text(
                f"You: _{transcript}_\n\n{response[:2000]}",
                parse_mode="Markdown",
            )
        except Exception:
            await placeholder.edit_text(
                f"You: {transcript}\n\n{response[:2000]}"
            )

        # Background: auto-capture
        if AUTO_CAPTURE:
            asyncio.create_task(auto_capture(transcript, response))

    except Exception as e:
        log.error("Voice handler error: %s", e)
        await placeholder.edit_text(f"Voice error: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("Starting Berryclaw...")
    log.info("Bot token: %s...%s", BOT_TOKEN[:8], BOT_TOKEN[-4:])
    log.info("Ollama URL: %s", OLLAMA_URL)
    log.info("Default model: %s", DEFAULT_MODEL)
    log.info("OpenRouter model: %s", OPENROUTER_MODEL)

    # Prune old messages on startup
    prune_old_messages()

    app = Application.builder().token(BOT_TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("model", cmd_model))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("api", cmd_api))
    app.add_handler(CommandHandler("think", cmd_think))
    app.add_handler(CommandHandler("identity", cmd_workspace))
    app.add_handler(CommandHandler("soul", cmd_workspace))
    app.add_handler(CommandHandler("user", cmd_workspace))
    app.add_handler(CommandHandler("modelx", cmd_modelx))
    app.add_handler(CommandHandler("remember", cmd_remember))
    app.add_handler(CommandHandler("memory", cmd_memory))
    app.add_handler(CommandHandler("forget", cmd_forget))
    app.add_handler(CommandHandler("profile", cmd_profile))
    app.add_handler(CommandHandler("skills", cmd_skills))
    app.add_handler(CommandHandler("newskill", cmd_newskill))
    app.add_handler(CommandHandler("deleteskill", cmd_deleteskill))
    app.add_handler(CommandHandler("agents", cmd_workspace))
    app.add_handler(CommandHandler("heartbeat", cmd_workspace))

    # Power skills — multimodal
    app.add_handler(CommandHandler("imagine", cmd_imagine))
    app.add_handler(CommandHandler("see", cmd_see))
    app.add_handler(CommandHandler("search", cmd_search))
    app.add_handler(CommandHandler("read", cmd_read))
    app.add_handler(CommandHandler("voice", cmd_voice))

    # Skill commands — register each loaded skill
    for trigger_name in SKILLS:
        app.add_handler(CommandHandler(trigger_name, _skill_command_handler))

    # Integration commands — auto-discovered from integrations/
    for cmd_name in INTEGRATIONS:
        app.add_handler(CommandHandler(cmd_name, _integration_command_handler))

    app.add_handler(CallbackQueryHandler(callback_model, pattern=r"^m:"))
    app.add_handler(CallbackQueryHandler(callback_pull, pattern=r"^pull:"))
    app.add_handler(CallbackQueryHandler(callback_cloudmodel, pattern=r"^cx:"))
    app.add_handler(CallbackQueryHandler(callback_soul, pattern=r"^soul:"))
    app.add_handler(CallbackQueryHandler(callback_api, pattern=r"^api:"))
    app.add_handler(CallbackQueryHandler(
        lambda u, c: log.warning("Unhandled callback: %s", u.callback_query.data)
    ))

    # Photos with /see caption
    app.add_handler(MessageHandler(
        filters.PHOTO & filters.CaptionRegex(r"^/see"),
        cmd_see,
    ))
    # Documents with /read caption
    app.add_handler(MessageHandler(
        filters.Document.ALL & filters.CaptionRegex(r"^/read"),
        cmd_read,
    ))

    # Voice messages — auto voice-in/voice-out with Deepgram
    app.add_handler(MessageHandler(filters.VOICE, handle_voice_message))

    # Regular messages
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    async def post_init(application):
        global _bot_app
        _bot_app = application
        asyncio.create_task(warmup_loop())
        asyncio.create_task(heartbeat_loop())

    app.post_init = post_init

    log.info("Bot is running. Press Ctrl+C to stop.")
    app.run_polling(
        drop_pending_updates=True,
        allowed_updates=["message", "callback_query"],
    )


if __name__ == "__main__":
    main()
