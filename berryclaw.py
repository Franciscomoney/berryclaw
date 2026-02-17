#!/usr/bin/env python3
"""Berryclaw — Lightweight Telegram bot for Raspberry Pi 5 + Ollama."""

import asyncio
import json
import logging
import os
import sqlite3
import time
from pathlib import Path

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
WORKSPACE_DIR = BASE_DIR / "workspace"

with open(CONFIG_PATH) as f:
    CFG = json.load(f)

BOT_TOKEN = CFG["telegram_bot_token"]
OLLAMA_URL = CFG["ollama_url"].rstrip("/")
DEFAULT_MODEL = CFG["default_model"]
MAX_HISTORY = CFG.get("max_history", 10)
STREAM_BATCH = CFG.get("stream_batch_tokens", 15)
WARMUP_INTERVAL = CFG.get("warmup_interval_seconds", 240)
ALLOWED_USERS: list[int] = CFG.get("allowed_users", [])
ADMIN_USERS: list[int] = CFG.get("admin_users", [])
OPENROUTER_KEY = CFG.get("openrouter_api_key", "")
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


def build_system_prompt_local() -> str:
    """Short system prompt for small Ollama models (< 100 tokens)."""
    identity = _read_workspace("IDENTITY.md")
    # Extract just the name and vibe from IDENTITY.md
    name = "Berryclaw"
    vibe = ""
    for line in identity.splitlines():
        if line.startswith("**Name:**"):
            name = line.split("**Name:**")[1].strip()
        elif line.startswith("**Vibe:**"):
            vibe = line.split("**Vibe:**")[1].strip()

    return (
        f"You are {name}. {vibe} "
        "Be helpful and brief. Answer directly, no filler. "
        "Never output XML, tool calls, or thinking tags."
    )


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
MEMORY_MODEL = CFG.get("memory_model", "google/gemini-2.0-flash-001")
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
    model = get_user_model(update.effective_chat.id)
    await update.message.reply_text(
        f"🫐 *Berryclaw* is online!\n\n"
        f"Running on Raspberry Pi 5 with local AI.\n"
        f"Current model: `{model}`\n\n"
        f"Commands: /help",
        parse_mode="Markdown",
    )


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
        "*Chat:*\n"
        "/model — Switch local AI model\n"
        "/modelx — Switch cloud AI model\n"
        "/think <query> — Use the cloud brain\n"
        "/skills — List available skills\n"
        "/clear — Forget our conversation\n\n"
        "*Memory (auto-learns from /think):*\n"
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

    # /identity (no args) — show current content
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
        await update.message.reply_text("Could not reach Ollama. Is it running?")
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

    # Build conversation
    history = get_history(chat_id)
    messages = [{"role": "system", "content": SYSTEM_PROMPT_LOCAL}]
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

    # Skill commands — register each loaded skill
    for trigger_name in SKILLS:
        app.add_handler(CommandHandler(trigger_name, _skill_command_handler))

    app.add_handler(CallbackQueryHandler(callback_model, pattern=r"^m:"))
    app.add_handler(CallbackQueryHandler(callback_cloudmodel, pattern=r"^cx:"))
    app.add_handler(CallbackQueryHandler(
        lambda u, c: log.warning("Unhandled callback: %s", u.callback_query.data)
    ))

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
