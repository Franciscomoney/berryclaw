#!/usr/bin/env python3
"""Berryclaw — Lightweight Telegram bot for Raspberry Pi 5 + Ollama."""

import asyncio
import json
import logging
import os
import re
import sqlite3
import time
from pathlib import Path

import base64
import io

import httpx
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    ApplicationHandlerStop,
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
OLLAMA_URL = (os.environ.get("OLLAMA_HOST")
              or CFG.get("ollama_url", "http://localhost:11434")).rstrip("/")
DEFAULT_MODEL = CFG.get("default_model", "qwen25-pi")
MAX_HISTORY = CFG.get("max_history", 10)
STREAM_BATCH = CFG.get("stream_batch_tokens", 15)
WARMUP_INTERVAL = CFG.get("warmup_interval_seconds", 240)
ALLOWED_USERS: list[int] = CFG.get("allowed_users", [])
ADMIN_USERS: list[int] = CFG.get("admin_users", [])
OPENROUTER_KEY = get_secret("openrouter_api_key")
OPENROUTER_MODEL = CFG.get("openrouter_model", "x-ai/grok-4.1-fast")
CLOUD_ONLY = CFG.get("cloud_only", False)  # Skip Ollama, use OpenRouter for all chat

# Group routing — map group IDs to workspace subdirectories
# Each group gets its own IDENTITY.md, SOUL.md in workspace/groups/<name>/
GROUP_ROUTING: dict[str, str] = CFG.get("group_routing", {})
# e.g. {"-1003812395835": "zote", "-1003848357518": "oracle"}

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

def _resolve_agent(chat_id: int = 0) -> str:
    """Resolve which agent handles a given chat_id.
    Returns agent name ('main', 'zote', 'oracle') or 'main' as default."""
    return GROUP_ROUTING.get(str(chat_id), "main") if chat_id else "main"


def _agent_workspace(agent: str) -> Path:
    """Get the workspace directory for an agent."""
    p = WORKSPACE_DIR / agent
    if p.is_dir():
        return p
    return WORKSPACE_DIR  # fallback to root workspace


def _read_workspace(filename: str, agent: str = "") -> str:
    """Read a workspace file, optionally from a specific agent's workspace."""
    if agent:
        p = _agent_workspace(agent) / filename
        if p.exists():
            return p.read_text().strip()
    # Fallback to root workspace
    p = WORKSPACE_DIR / filename
    return p.read_text().strip() if p.exists() else ""


PROFILE_PATH = WORKSPACE_DIR / "PROFILE.md"  # Legacy global path
MEMORY_DIR = WORKSPACE_DIR / "memory"


def _user_memory_dir(user_id: int, agent: str = "") -> Path:
    """Get or create per-user memory directory, scoped to agent workspace."""
    if agent:
        d = _agent_workspace(agent) / "memory" / str(user_id)
    else:
        d = MEMORY_DIR / str(user_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_profile(user_id: int = 0) -> str:
    """Read user profile. Per-user if user_id given, else legacy global."""
    if user_id:
        p = _user_memory_dir(user_id) / "PROFILE.md"
        if p.exists():
            return p.read_text().strip()
        # Migrate: check legacy global file
        if PROFILE_PATH.exists():
            content = PROFILE_PATH.read_text().strip()
            if content and "(empty" not in content:
                p.write_text(content + "\n")
                return content
        return ""
    # Legacy fallback
    if PROFILE_PATH.exists():
        return PROFILE_PATH.read_text().strip()
    return ""


def write_profile(content: str, user_id: int = 0):
    """Write user profile."""
    if user_id:
        p = _user_memory_dir(user_id) / "PROFILE.md"
        p.write_text(content.strip() + "\n")
    else:
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


def build_system_prompt_cloud(user_id: int = 0, chat_id: int = 0) -> str:
    """Full system prompt for cloud models — fully agent-scoped.

    Each agent loads ONLY its own workspace files. Main agent additionally
    gets a summary of subagent workspaces for cross-agent awareness.
    """
    agent = _resolve_agent(chat_id)

    parts = []
    for fname in ("IDENTITY.md", "SOUL.md", "USER.md", "AGENTS.md"):
        text = _read_workspace(fname, agent)
        if text:
            parts.append(text)

    # Agent-scoped user profile
    profile = read_profile(user_id)
    if profile:
        parts.append(f"# User Profile\n\n{profile}")

    # Main agent gets cross-agent visibility
    if agent == "main":
        subagent_summary = _build_subagent_summary()
        if subagent_summary:
            parts.append(subagent_summary)

    parts.append(f"# Context\nYou are **{agent.capitalize()}**. Agent mode: {'subagent' if agent != 'main' else 'commander'}.")

    return "\n\n---\n\n".join(parts)


def _build_subagent_summary() -> str:
    """Build a summary of all subagent workspaces for the Main agent."""
    summary_parts = ["# Subagent Status"]

    for agent_name in GROUP_ROUTING.values():
        if agent_name == "main":
            continue
        ws = _agent_workspace(agent_name)
        if not ws.is_dir():
            continue

        # Read identity
        identity = ""
        id_path = ws / "IDENTITY.md"
        if id_path.exists():
            identity = id_path.read_text().strip()

        # Read recent memory entries
        mem_dir = ws / "memory"
        recent_memory = ""
        if mem_dir.is_dir():
            # Get all memory files, newest first
            mem_files = sorted(mem_dir.glob("**/MEMORY.md"), reverse=True)
            for mf in mem_files[:3]:
                content = mf.read_text().strip()
                if content:
                    recent_memory += content[:500] + "\n"

        section = f"\n## {agent_name.capitalize()}\n"
        if identity:
            section += identity + "\n"
        if recent_memory:
            section += f"\n**Recent memory:**\n{recent_memory[:800]}\n"
        else:
            section += "\n*No memories recorded yet.*\n"

        summary_parts.append(section)

    return "\n".join(summary_parts) if len(summary_parts) > 1 else ""


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

MEMORY_PATH = WORKSPACE_DIR / "MEMORY.md"  # Legacy global path
MEMORY_MODEL = CFG.get("memory_model", "liquid/lfm-2.5-1.2b-instruct:free")
AUTO_CAPTURE = CFG.get("auto_capture", True)
PROFILE_FREQUENCY = CFG.get("profile_frequency", 20)  # Update profile every N /think calls
_think_counter: int = 0


def _user_memory_path(user_id: int, agent: str = "") -> Path:
    """Get per-user MEMORY.md path, scoped to agent workspace."""
    return _user_memory_dir(user_id, agent) / "MEMORY.md"


def read_memory(user_id: int = 0, chat_id: int = 0) -> str:
    """Read user memory, scoped to the agent for this chat."""
    agent = _resolve_agent(chat_id) if chat_id else ""
    if user_id:
        p = _user_memory_path(user_id, agent)
        if p.exists():
            return p.read_text().strip()
        # Migrate: check legacy paths
        legacy = _user_memory_path(user_id, "")
        if legacy.exists() and legacy != p:
            content = legacy.read_text().strip()
            if content and "(empty)" not in content:
                return content
        return ""
    # Legacy fallback
    return _read_workspace("MEMORY.md", agent)


def append_memory(note: str, user_id: int = 0, chat_id: int = 0):
    """Append a note to memory, scoped to the agent for this chat."""
    agent = _resolve_agent(chat_id) if chat_id else ""
    if user_id:
        p = _user_memory_path(user_id, agent)
        current = p.read_text() if p.exists() else ""
    else:
        current = MEMORY_PATH.read_text() if MEMORY_PATH.exists() else ""
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    current += f"\n- [{timestamp}] {note}"
    if user_id:
        _user_memory_dir(user_id, agent)  # ensure dir exists
        _user_memory_path(user_id, agent).write_text(current)
    else:
        MEMORY_PATH.write_text(current)


def clear_memory(user_id: int = 0, chat_id: int = 0):
    """Reset memory to empty, scoped to agent."""
    agent = _resolve_agent(chat_id) if chat_id else ""
    empty = "# Berryclaw Memory\n\n(empty)\n"
    if user_id:
        _user_memory_path(user_id, agent).write_text(empty)
    else:
        MEMORY_PATH.write_text(empty)
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
    conn.execute(
        """CREATE TABLE IF NOT EXISTS build_mode (
            chat_id INTEGER PRIMARY KEY,
            model TEXT NOT NULL
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS user_voice (
            chat_id INTEGER PRIMARY KEY,
            voice TEXT NOT NULL
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


DEFAULT_TTS_VOICE = "aura-asteria-en"
TTS_VOICES = {
    "asteria": ("aura-asteria-en", "Female, warm"),
    "luna": ("aura-luna-en", "Female, soft"),
    "stella": ("aura-stella-en", "Female, clear"),
    "athena": ("aura-athena-en", "Female, professional"),
    "hera": ("aura-hera-en", "Female, authoritative"),
    "orion": ("aura-orion-en", "Male, deep"),
    "arcas": ("aura-arcas-en", "Male, conversational"),
    "perseus": ("aura-perseus-en", "Male, warm"),
    "angus": ("aura-angus-en", "Male, Irish"),
    "orpheus": ("aura-orpheus-en", "Male, rich"),
    "helios": ("aura-helios-en", "Male, British"),
    "zeus": ("aura-zeus-en", "Male, authoritative"),
}


def get_user_voice(chat_id: int) -> str:
    row = DB.execute(
        "SELECT voice FROM user_voice WHERE chat_id = ?", (chat_id,)
    ).fetchone()
    return row[0] if row else DEFAULT_TTS_VOICE


def set_user_voice(chat_id: int, voice: str):
    DB.execute(
        "INSERT OR REPLACE INTO user_voice (chat_id, voice) VALUES (?, ?)",
        (chat_id, voice),
    )
    DB.commit()


def get_build_mode(chat_id: int) -> str | None:
    """Return the cloud model name if in build mode, else None."""
    row = DB.execute(
        "SELECT model FROM build_mode WHERE chat_id = ?", (chat_id,)
    ).fetchone()
    return row[0] if row else None


def set_build_mode(chat_id: int, model: str):
    DB.execute(
        "INSERT OR REPLACE INTO build_mode (chat_id, model) VALUES (?, ?)",
        (chat_id, model),
    )
    DB.commit()


def exit_build_mode(chat_id: int):
    DB.execute("DELETE FROM build_mode WHERE chat_id = ?", (chat_id,))
    DB.commit()


# ---------------------------------------------------------------------------
# Build Mode — Claude Code via tmux bridge
# ---------------------------------------------------------------------------

TMUX_SESSION = "claude-build"
PENDING_FILE = Path.home() / ".claude" / "telegram_pending"
CHAT_ID_FILE = Path.home() / ".claude" / "telegram_chat_id"
_build_polling_cancel: asyncio.Event | None = None  # Signal to stop current polling loop
_api_awaiting_key: dict[int, str] = {}  # {chat_id: key_name} — waiting for user to paste API key
_gauth_awaiting_code: dict[int, bool] = {}  # {chat_id: True} — waiting for user to paste Google auth code


def _tmux_exists() -> bool:
    import subprocess
    return subprocess.run(
        ["tmux", "has-session", "-t", TMUX_SESSION],
        capture_output=True,
    ).returncode == 0


def _tmux_send(text: str, literal: bool = True):
    import subprocess
    cmd = ["tmux", "send-keys", "-t", TMUX_SESSION]
    if literal:
        cmd.append("-l")
    cmd.append(text)
    subprocess.run(cmd)


def _tmux_send_enter():
    import subprocess
    subprocess.run(["tmux", "send-keys", "-t", TMUX_SESSION, "Enter"])


def _tmux_send_escape():
    import subprocess
    subprocess.run(["tmux", "send-keys", "-t", TMUX_SESSION, "Escape"])


def _start_claude_tmux(model: str):
    """Start Claude Code in a tmux session with the given model."""
    import subprocess
    # Kill existing session if any
    subprocess.run(["tmux", "kill-session", "-t", TMUX_SESSION], capture_output=True)
    time.sleep(0.5)

    env_str = (
        f'export ANTHROPIC_AUTH_TOKEN=ollama && '
        f'export ANTHROPIC_BASE_URL={OLLAMA_URL} && '
        f'export ANTHROPIC_API_KEY="" && '
        f'export PATH="$HOME/.local/bin:$PATH" && '
        f'cd ~/projects && '
        f'claude --model {model} --dangerously-skip-permissions'
    )
    subprocess.run([
        "tmux", "new-session", "-d", "-s", TMUX_SESSION,
        "bash", "-c", env_str,
    ])


def _stop_claude_tmux():
    """Kill the Claude Code tmux session."""
    import subprocess
    subprocess.run(["tmux", "kill-session", "-t", TMUX_SESSION], capture_output=True)


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
# Rate limiting — per-user, two tiers
# ---------------------------------------------------------------------------

# Limits: (max_requests, window_seconds)
RATE_LIMITS: dict[str, tuple[int, int]] = {
    "chat":  (CFG.get("rate_limit_chat", 30),  60),   # 30 msgs/min local chat
    "cloud": (CFG.get("rate_limit_cloud", 10),  60),   # 10 reqs/min cloud commands
}

_rate_buckets: dict[tuple[int, str], list[float]] = {}  # (user_id, tier) → [timestamps]


def _rate_check(user_id: int, tier: str = "chat") -> str | None:
    """Check rate limit. Returns None if OK, or a message string if limited. Admins are exempt."""
    if is_admin(user_id):
        return None
    max_req, window = RATE_LIMITS.get(tier, (30, 60))
    if max_req <= 0:
        return None  # 0 = disabled
    key = (user_id, tier)
    now = time.time()
    timestamps = _rate_buckets.get(key, [])
    # Prune old entries
    timestamps = [t for t in timestamps if now - t < window]
    if len(timestamps) >= max_req:
        wait = int(window - (now - timestamps[0])) + 1
        return f"⏳ Slow down — limit is {max_req} per minute.\n\nTry again in {wait}s."
    timestamps.append(now)
    _rate_buckets[key] = timestamps
    return None


def _is_group_chat(update: Update) -> bool:
    """Check if message is from a group/supergroup."""
    return update.effective_chat.type in ("group", "supergroup")


def _bot_mentioned(update: Update) -> bool:
    """Check if bot was @mentioned or replied to in a group message."""
    msg = update.message
    if not msg:
        return False
    # Direct reply to the bot
    if msg.reply_to_message and msg.reply_to_message.from_user:
        if msg.reply_to_message.from_user.is_bot and _bot_username:
            if (msg.reply_to_message.from_user.username or "").lower() == _bot_username:
                return True
    # @mentioned in text
    text = msg.text or msg.caption or ""
    if _bot_username and f"@{_bot_username}" in text.lower():
        return True
    # Check entities for mention
    for ent in msg.entities or []:
        if ent.type == "mention":
            mention = text[ent.offset:ent.offset + ent.length].lower()
            if mention == f"@{_bot_username}":
                return True
    return False


def _strip_mention(text: str) -> str:
    """Remove @botname from message text."""
    if _bot_username:
        import re
        text = re.sub(rf"@{re.escape(_bot_username)}\b", "", text, flags=re.IGNORECASE).strip()
    return text


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
    """Get available Ollama models (local + cloud)."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            r.raise_for_status()
            data = r.json()
            return [m["name"] for m in data.get("models", [])]
    except Exception as e:
        log.error("Failed to list models: %s", e)
        return []


# Popular cloud models — shown as pull options in /model
CLOUD_CATALOG = [
    ("minimax-m2.5:cloud", "Fast coding & productivity"),
    ("deepseek-v3.1:cloud", "671B reasoning powerhouse"),
    ("qwen3-coder-next:cloud", "Agentic coding specialist"),
    ("glm-5:cloud", "744B systems engineering"),
    ("glm-4.7-flash:cloud", "Fast GLM flash model"),
    ("qwen3.5:cloud", "397B hybrid vision-language"),
    ("kimi-k2.5:cloud", "Multimodal agentic"),
]

# Recommended local models for first-time setup (shown when Ollama has 0 models)
RECOMMENDED_LOCAL_MODELS = [
    ("huihui_ai/qwen2.5-abliterate:1.5b", "Best quality", "~1 GB"),
    ("huihui_ai/gemma3-abliterated:1b", "Good & lighter", "~770 MB"),
    ("huihui_ai/qwen3-abliterated:0.6b", "Fastest, basic", "~380 MB"),
]


def _is_cloud_model(name: str) -> bool:
    """Check if a model name is a cloud model."""
    return ":cloud" in name.lower()


# ---------------------------------------------------------------------------
# OpenRouter — cloud escalation for /think
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(
    r"<[a-zA-Z_:]+tool_call>.*?</[a-zA-Z_:]+tool_call>",
    re.DOTALL,
)


def _sanitize_model_output(text: str) -> str:
    """Strip hallucinated tool call XML and other junk from model responses."""
    if not text:
        return text
    # Remove <minimax:tool_call>...</minimax:tool_call> and similar patterns
    cleaned = _TOOL_CALL_RE.sub("", text)
    # Remove <tool_call>...</tool_call> variants
    cleaned = re.sub(r"</?tool_call>", "", cleaned)
    # Remove <invoke>...</invoke> blocks
    cleaned = re.sub(r"<invoke\b.*?</invoke>", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


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
            content = data["choices"][0]["message"]["content"]
            return _sanitize_model_output(content)
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


def _friendly_error(error: str | Exception, context: str = "") -> str:
    """Turn raw errors into user-friendly Telegram messages."""
    err = str(error).lower()

    # OpenRouter / API key issues
    if "api key not configured" in err:
        return "🔑 No OpenRouter API key set.\n\nAdd one via /api to enable cloud features."
    if "401" in err or "unauthorized" in err or "invalid api key" in err:
        return "🔑 Your OpenRouter API key is invalid or expired.\n\nUpdate it via /api."
    if "402" in err or "payment required" in err or "insufficient credits" in err:
        return "💳 OpenRouter credits ran out.\n\nTop up at openrouter.ai/credits."
    if "429" in err or "rate limit" in err or "too many requests" in err:
        return "⏳ Rate limited — too many requests.\n\nWait a moment and try again."
    if "model not found" in err or "no endpoints" in err or "does not exist" in err:
        return f"🤖 Model not available.\n\nTry switching models with /modelx."

    # Ollama issues
    if "connect" in err and ("refused" in err or "11434" in err or "ollama" in err):
        return "🔌 Can't reach Ollama.\n\nMake sure it's running: `ollama serve`"
    if "timeout" in err or "timed out" in err:
        return "⏱ Request timed out.\n\nThe model took too long. Try again or use a smaller model."

    # Network issues
    if "network" in err or "dns" in err or "unreachable" in err:
        return "🌐 Network error — can't reach the internet.\n\nCheck your connection."

    # Deepgram
    if "deepgram" in err or "dgram" in err:
        return "🎤 Voice service error.\n\nCheck your Deepgram API key via /api."

    # Generic — keep it short, hide the traceback
    prefix = f"{context} error" if context else "Something went wrong"
    short = str(error)[:150]
    return f"❌ {prefix}.\n\n`{short}`"


# ---------------------------------------------------------------------------
# Smart Memory — auto-capture, smart recall, profile building
# ---------------------------------------------------------------------------


async def auto_capture(user_msg: str, assistant_msg: str, user_id: int = 0, chat_id: int = 0):
    """Extract key facts from a conversation turn and save to agent-scoped memory.
    Runs in background after /think responses. Uses a cheap fast model."""
    try:
        existing = read_memory(user_id, chat_id)
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
                f"Existing memory:\n{existing}\n\n"
                f"---\n\nNew conversation:\n"
                f"User: {user_msg}\n"
                f"Assistant: {assistant_msg[:1000]}"
            )},
        ]
        result = await openrouter_chat(messages, model=MEMORY_MODEL)
        if result and "NOTHING" not in result.upper():
            for line in result.strip().splitlines():
                line = line.strip()
                if line.startswith("- "):
                    append_memory(line[2:], user_id, chat_id)
            agent = _resolve_agent(chat_id)
            log.info("Auto-capture: saved facts for user %s (agent: %s)", user_id, agent)
    except Exception as e:
        log.warning("Auto-capture failed: %s", e)


async def smart_recall(query: str, user_id: int = 0, chat_id: int = 0) -> str:
    """Given a user query, return only the relevant memories from this agent's scope.
    Uses a cheap fast model to filter. Returns empty string if no relevant memories."""
    memory = read_memory(user_id, chat_id)
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


async def update_profile(chat_id: int, user_id: int = 0):
    """Rebuild user profile from conversation history and memory.
    Called every PROFILE_FREQUENCY /think calls."""
    try:
        history = get_history(chat_id)
        memory = read_memory(user_id, chat_id)
        current_profile = read_profile(user_id)

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
            write_profile(result, user_id)
            reload_prompts()
            log.info("Profile updated for user %s (%d chars)", user_id, len(result))
    except Exception as e:
        log.warning("Profile update failed: %s", e)


# ---------------------------------------------------------------------------
# Warmup — keep model loaded in memory
# ---------------------------------------------------------------------------

_last_used_model: str = DEFAULT_MODEL
HEARTBEAT_INTERVAL = CFG.get("heartbeat_interval_seconds", 1800)  # 30 min
_bot_app = None  # Set in main() for heartbeat messaging
_bot_username: str = ""  # Set in post_init for @mention detection


async def warmup_loop():
    """Ping Ollama every WARMUP_INTERVAL seconds to prevent cold starts."""
    if WARMUP_INTERVAL <= 0 or CLOUD_ONLY:
        log.info("Warmup disabled%s", " (cloud-only mode)" if CLOUD_ONLY else "")
        return
    log.info("Warmup loop started (every %ds)", WARMUP_INTERVAL)
    while True:
        await asyncio.sleep(WARMUP_INTERVAL)
        # Skip warmup for cloud models — they don't need local RAM
        if _is_cloud_model(_last_used_model):
            log.debug("Skipping warmup for cloud model %s", _last_used_model)
            continue
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

    # Auto-admin: if no admins configured, first /start user becomes admin
    if not ADMIN_USERS:
        user_id = update.effective_user.id
        ADMIN_USERS.append(user_id)
        CFG["admin_users"] = ADMIN_USERS
        with open(CONFIG_PATH, "w") as f:
            json.dump(CFG, f, indent=2)
            f.write("\n")
        log.info("Auto-admin: user %s is now admin", user_id)
        await update.message.reply_text(
            f"👑 You're the first user — you are now admin!\n"
            f"Your user ID: `{user_id}`",
            parse_mode="Markdown",
        )

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

    # Show recommended model buttons if Ollama is up but no models
    buttons = []
    if models == [] and not any("not reachable" in i for i in issues):
        for name, desc, size in RECOMMENDED_LOCAL_MODELS:
            buttons.append([InlineKeyboardButton(
                f"📥 {name.split('/')[-1]} — {desc} ({size})",
                callback_data=f"pull:{name}",
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
        "/voice — Pick TTS voice or transcribe a voice message\n"
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
        "/help — This message\n\n"
        "*Build Mode (Claude Code):*\n"
        "/build — Start Claude Code with a cloud model\n"
        "/stop — Interrupt Claude Code\n"
        "/exit — Exit Build Mode\n"
        "/claude — View/add Build Mode rules\n"
        "/auth — View/change project login\n"
        "/api — Manage API keys (tap to set)",
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


CLAUDE_MD_PATH = Path.home() / "projects" / "CLAUDE.md"
AUTH_FILE_PATH = Path.home() / "projects" / ".auth"
DEFAULT_AUTH = {"username": "admin", "password": "berryclaw"}


def _read_auth() -> dict:
    """Read auth credentials from ~/projects/.auth."""
    if AUTH_FILE_PATH.exists():
        import json as _json
        try:
            return _json.loads(AUTH_FILE_PATH.read_text())
        except Exception:
            pass
    return DEFAULT_AUTH.copy()


def _write_auth(username: str, password: str):
    """Write auth credentials to ~/projects/.auth."""
    import json as _json
    AUTH_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUTH_FILE_PATH.write_text(_json.dumps({"username": username, "password": password}))


async def cmd_auth(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """View or change the login credentials for web projects."""
    if not is_allowed(update.effective_user.id):
        return

    args = update.message.text.split(maxsplit=2)

    # /auth (no args) — show current credentials
    if len(args) < 2:
        creds = _read_auth()
        await update.message.reply_text(
            f"🔐 *Project Auth*\n\n"
            f"Username: `{creds['username']}`\n"
            f"Password: `{creds['password']}`\n\n"
            "All web projects use these credentials.\n"
            "Change: `/auth <username> <password>`\n"
            "Reset: `/auth reset`",
            parse_mode="Markdown",
        )
        return

    text = args[1].strip()

    # /auth reset — restore defaults
    if text.lower() == "reset":
        _write_auth(DEFAULT_AUTH["username"], DEFAULT_AUTH["password"])
        await update.message.reply_text(
            f"🔄 Auth reset to `{DEFAULT_AUTH['username']}`/`{DEFAULT_AUTH['password']}`",
            parse_mode="Markdown",
        )
        return

    # /auth <username> <password>
    if len(args) < 3:
        await update.message.reply_text(
            "Usage: `/auth <username> <password>`",
            parse_mode="Markdown",
        )
        return

    new_user = args[1].strip()
    new_pass = args[2].strip()
    _write_auth(new_user, new_pass)
    await update.message.reply_text(
        f"✅ Auth updated\n\n"
        f"Username: `{new_user}`\n"
        f"Password: `{new_pass}`\n\n"
        "New projects will use these credentials.\n"
        "Running projects need a restart to pick up the change.",
        parse_mode="Markdown",
    )


async def cmd_claude(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """View or add rules to the Build Mode CLAUDE.md."""
    if not is_allowed(update.effective_user.id):
        return

    args = update.message.text.split(maxsplit=1)

    # /claude (no args) — show current content
    if len(args) < 2:
        content = CLAUDE_MD_PATH.read_text() if CLAUDE_MD_PATH.exists() else "(empty)"
        if len(content) > 3900:
            content = content[:3900] + "\n\n... (truncated)"
        await update.message.reply_text(
            f"📋 *CLAUDE.md* (Build Mode rules)\n\n```\n{content}\n```\n\n"
            "Add a rule: `/claude <rule>`\n"
            "Reset: `/claude reset`",
            parse_mode="Markdown",
        )
        return

    text = args[1].strip()

    # /claude reset — restore to default
    if text.lower() == "reset":
        # Re-copy from raspberryclaw repo if it exists
        default = Path.home() / "raspberryclaw" / "CLAUDE.md"
        if default.exists():
            CLAUDE_MD_PATH.write_text(default.read_text())
            await update.message.reply_text("🔄 CLAUDE.md reset to defaults.")
        else:
            await update.message.reply_text("❌ No default CLAUDE.md found.")
        return

    # /claude <rule> — append a custom rule
    CLAUDE_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CLAUDE_MD_PATH, "a") as f:
        f.write(f"\n## Custom Rule\n\n{text}\n")

    await update.message.reply_text(
        f"✅ Rule added to CLAUDE.md.\n\nClaude Code will follow it next time you `/build`.",
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
    cid = update.effective_chat.id
    append_memory(note, update.effective_user.id, cid)
    agent = _resolve_agent(cid)
    await update.message.reply_text(f"Saved to {agent}'s memory.")


async def cmd_memory(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Show long-term memory for this agent."""
    if not is_allowed(update.effective_user.id):
        return
    cid = update.effective_chat.id
    agent = _resolve_agent(cid)
    content = read_memory(update.effective_user.id, cid)
    if not content or "(empty)" in content:
        await update.message.reply_text(f"{agent}'s memory is empty. Use `/remember <note>` to add.", parse_mode="Markdown")
        return
    if len(content) > 3900:
        content = content[:3900] + "\n\n... (truncated)"
    await update.message.reply_text(f"*{agent.capitalize()}'s Memory*\n\n```\n{content}\n```", parse_mode="Markdown")


async def cmd_forget(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Clear all long-term memory for this agent."""
    if not is_allowed(update.effective_user.id):
        return
    cid = update.effective_chat.id
    agent = _resolve_agent(cid)
    clear_memory(update.effective_user.id, cid)
    await update.message.reply_text(f"{agent.capitalize()}'s memory cleared.")


async def cmd_profile(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """View or reset user profile."""
    if not is_allowed(update.effective_user.id):
        return

    args = update.message.text.split(maxsplit=1)

    user_id = update.effective_user.id

    # /profile reset — clear profile
    if len(args) > 1 and args[1].strip().lower() == "reset":
        write_profile("# User Profile\n\n(empty — will build automatically)", user_id)
        reload_prompts()
        await update.message.reply_text("Profile reset.")
        return

    # /profile — show it
    profile = read_profile(user_id)
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

        # Support dict returns with reply_markup for inline buttons
        if isinstance(result, dict):
            text = result.get("text", "(no result)")
            markup = result.get("reply_markup")
            if len(text) > 4000:
                text = text[:4000] + "\n\n... (truncated)"
            try:
                await placeholder.edit_text(text, parse_mode="Markdown", reply_markup=markup)
            except Exception:
                await placeholder.edit_text(text, reply_markup=markup)
        else:
            if len(result) > 4000:
                result = result[:4000] + "\n\n... (truncated)"
            try:
                await placeholder.edit_text(result, parse_mode="Markdown")
            except Exception:
                await placeholder.edit_text(result)

    except Exception as e:
        log.error("Integration %s error: %s", command, e)
        await placeholder.edit_text(_friendly_error(e, command))


async def callback_google(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Handle Google Sheets/Docs inline button callbacks (gs:xxx)."""
    query = update.callback_query
    await query.answer()
    data = query.data  # e.g. "gs:list", "gs:read", "gs:auth", "gd:read"

    integration = INTEGRATIONS.get("sheets") or INTEGRATIONS.get("docs") or INTEGRATIONS.get("gauth")
    if not integration:
        await query.edit_message_text("Google integration not loaded.")
        return

    async def cloud_chat(messages, model=None):
        return await openrouter_chat(messages, model=model)

    secrets = {k: get_secret(k) for k in SECRETS} if SECRETS else CFG

    try:
        if data == "gs:list":
            result = await integration["handle"](command="sheets", args="list", secrets=secrets, cloud_chat=cloud_chat)
        elif data == "gs:read":
            await query.edit_message_text("Send the sheet ID and range:\n`/sheets read <sheet_id> [A1:Z100]`", parse_mode="Markdown")
            return
        elif data == "gs:write":
            await query.edit_message_text("Send the write command:\n`/sheets write <sheet_id> <range> <value>`", parse_mode="Markdown")
            return
        elif data == "gs:append":
            await query.edit_message_text("Send the append command:\n`/sheets append <sheet_id> <range> val1 | val2 | val3`", parse_mode="Markdown")
            return
        elif data == "gs:auth":
            result = await integration["handle"](command="gauth", args="", secrets=secrets, cloud_chat=cloud_chat)
            # Set awaiting flag so next message is treated as the auth code
            cid = query.message.chat_id
            _gauth_awaiting_code[cid] = True
            # Move pending flow to correct chat_id (use integration's module instance)
            try:
                mod = integration.get("module")
                if mod and hasattr(mod, "_pending_flow"):
                    if 0 in mod._pending_flow:
                        mod._pending_flow[cid] = mod._pending_flow.pop(0)
            except Exception:
                pass
        elif data == "gd:read":
            await query.edit_message_text("Send the doc ID:\n`/docs read <doc_id>`", parse_mode="Markdown")
            return
        elif data == "gd:append":
            await query.edit_message_text("Send the append command:\n`/docs append <doc_id> <text>`", parse_mode="Markdown")
            return
        elif data == "gd:ask":
            await query.edit_message_text("Send your question:\n`/docs ask <doc_id> <question>`", parse_mode="Markdown")
            return
        else:
            await query.edit_message_text(f"Unknown action: {data}")
            return

        # Handle result (string or dict)
        if isinstance(result, dict):
            text = result.get("text", "(no result)")
            markup = result.get("reply_markup")
            try:
                await query.edit_message_text(text, parse_mode="Markdown", reply_markup=markup)
            except Exception:
                await query.edit_message_text(text, reply_markup=markup)
        else:
            if len(result) > 4000:
                result = result[:4000] + "\n\n... (truncated)"
            try:
                await query.edit_message_text(result, parse_mode="Markdown")
            except Exception:
                await query.edit_message_text(result)

    except Exception as e:
        log.error("Google callback error: %s", e)
        await query.edit_message_text(f"Error: {e}")


async def callback_leads(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Handle Lead Scraper inline button callbacks (leads:xxx)."""
    query = update.callback_query
    await query.answer()

    integration = INTEGRATIONS.get("leads")
    if not integration:
        await query.edit_message_text("Leads integration not loaded.")
        return

    mod = integration.get("module")
    if not mod or not hasattr(mod, "callback"):
        await query.edit_message_text("Leads callback not available.")
        return

    secrets = {k: get_secret(k) for k in SECRETS} if SECRETS else CFG

    try:
        await mod.callback(query, secrets)
    except Exception as e:
        log.error("Leads callback error: %s", e)
        try:
            await query.edit_message_text(f"Error: {e}")
        except Exception:
            pass


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
            # Ollama is up but no models — offer recommended ones
            buttons = []
            for name, desc, size in RECOMMENDED_LOCAL_MODELS:
                buttons.append([InlineKeyboardButton(
                    f"📥 {name.split('/')[-1]} — {desc} ({size})",
                    callback_data=f"pull:{name}",
                )])
            await update.message.reply_text(
                "📦 *No models installed yet.*\n\n"
                "Pick a model to download:\n"
                "_(recommended: qwen2.5-abliterate 1.5B)_",
                parse_mode="Markdown",
                reply_markup=InlineKeyboardMarkup(buttons),
            )
        except Exception:
            await update.message.reply_text(
                "❌ *Can't reach Ollama.*\n\n"
                "Make sure it's installed and running:\n"
                "`ollama serve`",
                parse_mode="Markdown",
            )
        return

    # Split into local and cloud models
    local_models = [m for m in available if not _is_cloud_model(m)]
    cloud_models = [m for m in available if _is_cloud_model(m)]

    buttons = []

    # Local models section
    if local_models:
        buttons.append([InlineKeyboardButton("— Local Models (on Pi) —", callback_data="m:_header")])
        for i, m in enumerate(local_models):
            label = f"✅ {m}" if m == current else m
            buttons.append([InlineKeyboardButton(label, callback_data=f"m:{i}:{m[:50]}")])

    # Cloud models already pulled
    if cloud_models:
        buttons.append([InlineKeyboardButton("— Cloud Models (Ollama) —", callback_data="m:_header")])
        for i, m in enumerate(cloud_models):
            label = f"✅ {m}" if m == current else m
            buttons.append([InlineKeyboardButton(
                label, callback_data=f"m:{len(local_models)+i}:{m[:50]}",
            )])

    # Cloud models available to pull (not yet installed)
    installed_names = {m.split(":")[0] for m in available}
    pullable = [(name, desc) for name, desc in CLOUD_CATALOG
                if name.split(":")[0] not in installed_names]
    if pullable and get_secret("ollama_api_key"):
        buttons.append([InlineKeyboardButton("— Add Cloud Model —", callback_data="m:_header")])
        for name, desc in pullable[:4]:
            buttons.append([InlineKeyboardButton(
                f"☁️ {name} — {desc}", callback_data=f"pull:{name}",
            )])

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

        # Update default_model in config if this is the first model
        if CFG.get("default_model", "") in ("", "qwen25-pi"):
            CFG["default_model"] = model_name
            with open(CONFIG_PATH, "w") as f:
                json.dump(CFG, f, indent=2)
                f.write("\n")
            log.info("Updated default_model to %s", model_name)

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
    """Show cloud model picker with inline buttons.

    Usage:
      /modelx           — pick model for current agent
      /modelx zote      — pick model for Zote (admin, from any chat)
      /modelx oracle    — pick model for Oracle (admin, from any chat)
      /modelx status    — show all agents' current models
    """
    if not is_allowed(update.effective_user.id):
        return

    args = update.message.text.split(maxsplit=1)
    arg = args[1].strip().lower() if len(args) > 1 else ""

    # /modelx status — show all agent models
    if arg == "status":
        lines = []
        # Current chat
        for agent_name, group_id in GROUP_ROUTING.items():
            model = get_user_cloud_model(int(group_id))
            lines.append(f"**{_agent_name_from_group(group_id)}**: `{model}`")
        # Main (DMs use chat_id of the user)
        main_model = OPENROUTER_MODEL  # default
        lines.insert(0, f"**Main** (default): `{main_model}`")
        await update.message.reply_text(
            "**Agent Models:**\n" + "\n".join(lines),
            parse_mode="Markdown",
        )
        return

    # /modelx <agent> — set model for a specific agent (admin only)
    target_chat_id = update.effective_chat.id
    target_label = _resolve_agent(target_chat_id).capitalize()

    if arg and arg in ("zote", "oracle", "main"):
        if not is_admin(update.effective_user.id):
            await update.message.reply_text("Only admins can change other agents' models.")
            return
        # Find the group chat_id for this agent
        for gid, aname in GROUP_ROUTING.items():
            if aname == arg:
                target_chat_id = int(gid)
                target_label = arg.capitalize()
                break
        else:
            if arg == "main":
                target_label = "Main"
                # Main uses the current DM chat_id
            else:
                await update.message.reply_text(f"Agent `{arg}` not found.")
                return

    current = get_user_cloud_model(target_chat_id)

    buttons = []
    for m in CLOUD_MODELS:
        label = f"✅ {m}" if m == current else m
        # Encode target chat_id in callback data
        buttons.append([InlineKeyboardButton(label, callback_data=f"cx:{target_chat_id}:{m}")])

    await update.message.reply_text(
        f"*Pick cloud model for {target_label}:*\nCurrent: `{current}`",
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(buttons),
    )


def _agent_name_from_group(group_id: str) -> str:
    """Get agent name from group ID."""
    return GROUP_ROUTING.get(str(group_id), "main").capitalize()


async def callback_cloudmodel(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Handle inline button press for cloud model selection.
    Callback data format: cx:<target_chat_id>:<model> or legacy cx:<model>"""
    query = update.callback_query

    if not is_allowed(query.from_user.id):
        await query.answer("Not authorized.")
        return

    data = query.data or ""
    if not data.startswith("cx:"):
        await query.answer()
        return

    parts = data.split(":", 2)
    if len(parts) == 3:
        # New format: cx:<target_chat_id>:<model>
        try:
            target_chat_id = int(parts[1])
        except ValueError:
            target_chat_id = query.message.chat_id
        chosen = parts[2]
    else:
        # Legacy format: cx:<model>
        target_chat_id = query.message.chat_id
        chosen = parts[1]

    set_user_cloud_model(target_chat_id, chosen)
    agent = _resolve_agent(target_chat_id)

    await query.answer(f"{agent.capitalize()}: {chosen}")
    await query.edit_message_text(
        f"**{agent.capitalize()}** model set to `{chosen}`",
        parse_mode="Markdown",
    )


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


async def callback_voice(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Handle voice picker button press."""
    query = update.callback_query
    if not is_allowed(query.from_user.id):
        await query.answer("Not authorized.")
        return

    name = (query.data or "").split(":", 1)[1]
    if name not in TTS_VOICES:
        await query.answer("Unknown voice.")
        return

    voice_id, desc = TTS_VOICES[name]
    chat_id = query.message.chat_id
    set_user_voice(chat_id, voice_id)
    await query.answer(f"Voice: {name}")

    # Rebuild buttons with new selection
    buttons = []
    row = []
    for n, (vid, d) in TTS_VOICES.items():
        label = f"{'✅ ' if vid == voice_id else ''}{n}"
        row.append(InlineKeyboardButton(label, callback_data=f"voice:{n}"))
        if len(row) == 3:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)

    await query.edit_message_text(
        f"🎙 *Voice Settings*\n\n"
        f"Current: *{name}* ({desc})\n\n"
        f"Send a voice note and I'll respond with voice!\n"
        f"Tap below to change voice:",
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(buttons),
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
        chat_id = query.message.chat_id
        _api_awaiting_key[chat_id] = key_name
        await query.answer()
        await query.edit_message_text(
            f"🔑 Paste your `{key_name}` below.\n\n"
            f"Just paste the key and hit send — it will be saved and your message auto-deleted.",
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


async def cmd_build(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Enter Build Mode — real Claude Code via tmux."""
    if not is_allowed(update.effective_user.id):
        return

    chat_id = update.effective_chat.id
    current_build = get_build_mode(chat_id)

    # Already in build mode?
    if current_build:
        alive = _tmux_exists()
        status = "running" if alive else "stopped"
        buttons = [[InlineKeyboardButton("🚪 Exit Build Mode", callback_data="build:exit")]]
        if not alive:
            buttons.insert(0, [InlineKeyboardButton("🔄 Restart session", callback_data=f"build:{current_build}")])
        await update.message.reply_text(
            f"☁️ *Build Mode active*\n\n"
            f"Model: `{current_build}`\n"
            f"Session: {status}\n\n"
            "Just type — everything goes to Claude Code.\n"
            "Type /exit when done.",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(buttons),
        )
        return

    # Show model picker
    available = await ollama_list_models()
    pulled_cloud = [m for m in available if _is_cloud_model(m)]

    buttons = []
    for m in pulled_cloud:
        buttons.append([InlineKeyboardButton(f"☁️ {m}", callback_data=f"build:{m}")])

    pulled_names = {m.split(":")[0] for m in pulled_cloud}
    for name, desc in CLOUD_CATALOG:
        if name.split(":")[0] not in pulled_names:
            buttons.append([InlineKeyboardButton(
                f"📥 {name} — {desc}", callback_data=f"build:{name}",
            )])

    await update.message.reply_text(
        "☁️ *Build Mode — Claude Code*\n\n"
        "Pick a cloud model. This starts a real Claude Code\n"
        "session on your Pi — it can read files, write code,\n"
        "and run commands.\n\n"
        "Type /exit when you're done.",
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(buttons),
    )


async def cmd_exit(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Exit Build Mode — kill tmux session, back to local chat."""
    global _build_polling_cancel, _build_menu_future
    if not is_allowed(update.effective_user.id):
        return

    chat_id = update.effective_chat.id
    build_model = get_build_mode(chat_id)

    if not build_model:
        await update.message.reply_text("You're not in Build Mode.")
        return

    # Cancel any active polling loop immediately
    if _build_polling_cancel is not None and not _build_polling_cancel.is_set():
        _build_polling_cancel.set()
    # Cancel any pending menu selection
    if _build_menu_future and not _build_menu_future.done():
        _build_menu_future.cancel()
        _build_menu_future = None

    _stop_claude_tmux()
    exit_build_mode(chat_id)
    # Clean up pending files
    if PENDING_FILE.exists():
        PENDING_FILE.unlink()

    local_model = get_user_model(chat_id)
    await update.message.reply_text(
        f"🏠 *Back to normal chat*\n\n"
        f"Model: `{local_model}`\n\n"
        "Type `/build` to start Claude Code again.",
        parse_mode="Markdown",
    )


async def cmd_stop(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Interrupt Claude Code — stop current task without exiting Build Mode."""
    global _build_polling_cancel
    if not is_allowed(update.effective_user.id):
        return

    chat_id = update.effective_chat.id
    build_model = get_build_mode(chat_id)

    if not build_model:
        await update.message.reply_text("You're not in Build Mode.")
        return

    # Cancel the polling loop
    if _build_polling_cancel is not None and not _build_polling_cancel.is_set():
        _build_polling_cancel.set()

    # Send Escape to interrupt Claude Code
    if _tmux_exists():
        _tmux_send_escape()

    await update.message.reply_text(
        "⚡ *Interrupted*\n\nClaude Code stopped. Send a new message or /exit.",
        parse_mode="Markdown",
    )


async def callback_build(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Handle /build button — start/exit Claude Code tmux session."""
    query = update.callback_query
    if not is_allowed(query.from_user.id):
        await query.answer("Not authorized.")
        return

    data = query.data or ""
    if not data.startswith("build:"):
        await query.answer()
        return

    choice = data[6:]  # After "build:"
    chat_id = query.message.chat_id

    # Exit build mode
    if choice == "exit":
        _stop_claude_tmux()
        exit_build_mode(chat_id)
        if PENDING_FILE.exists():
            PENDING_FILE.unlink()
        local_model = get_user_model(chat_id)
        await query.answer("Build Mode off")
        await query.edit_message_text(
            f"🏠 *Back to normal chat*\n\n"
            f"Model: `{local_model}`\n\n"
            "Type `/build` to start Claude Code again.",
            parse_mode="Markdown",
        )
        return

    # Cloud model selected — pull if needed, then start tmux session
    model_name = choice
    available = await ollama_list_models()

    if model_name not in available:
        # Need to pull first
        await query.answer("Setting up…")
        await query.edit_message_text(
            f"☁️ *Pulling `{model_name}`…*\n\nFirst time only.",
            parse_mode="Markdown",
        )
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                r = await client.post(
                    f"{OLLAMA_URL}/api/pull",
                    json={"name": model_name, "stream": False},
                )
                if r.status_code == 401 or "unauthorized" in r.text.lower():
                    import subprocess as sp, re
                    result = sp.run(
                        ["bash", "-c", "timeout 8 ollama signin 2>&1 || true"],
                        capture_output=True, text=True, timeout=15,
                        env={**__import__("os").environ, "TERM": "dumb"},
                    )
                    url_match = re.search(r"(https://ollama\.com/connect\S+)", result.stdout + result.stderr)
                    if url_match:
                        await query.edit_message_text(
                            "🔑 *Sign in to Ollama first:*\n\n"
                            f"[Sign in]({url_match.group(1)})\n\n"
                            "Then tap `/build` again.",
                            parse_mode="Markdown",
                            disable_web_page_preview=True,
                        )
                    else:
                        await query.edit_message_text("❌ Run `ollama signin` on your Pi first.", parse_mode="Markdown")
                    return
                r.raise_for_status()
        except Exception as e:
            await query.edit_message_text(_friendly_error(e, "Build Mode"))
            return

    # Start Claude Code in tmux
    await query.answer("Starting Claude Code…")
    await query.edit_message_text(
        f"☁️ *Starting Claude Code…*\n\nModel: `{model_name}`",
        parse_mode="Markdown",
    )

    _start_claude_tmux(model_name)
    await asyncio.sleep(3)

    if _tmux_exists():
        set_build_mode(chat_id, model_name)
        # Save chat_id for the stop hook
        CHAT_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
        CHAT_ID_FILE.write_text(str(chat_id))

        await query.edit_message_text(
            f"☁️ *Build Mode — Claude Code is running*\n\n"
            f"Model: `{model_name}`\n\n"
            "Type anything — it goes straight to Claude Code.\n"
            "Claude can read files, write code, run commands.\n\n"
            "Type /exit when you're done.",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("🚪 Exit Build Mode", callback_data="build:exit"),
            ]]),
        )
    else:
        await query.edit_message_text(
            "❌ Claude Code failed to start.\n\nCheck if it's installed: `claude --version`",
            parse_mode="Markdown",
        )


async def callback_build_menu(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Handle menu option selection from Build Mode interactive menus."""
    global _build_menu_future
    query = update.callback_query

    data = query.data or ""
    if not data.startswith("bmenu:"):
        return

    choice = int(data.split(":")[1])
    await query.answer(f"Option {choice + 1} selected")

    # Show what was selected
    try:
        buttons = query.message.reply_markup.inline_keyboard
        if choice < len(buttons):
            selected_text = buttons[choice][0].text
            await query.edit_message_text(f"✅ {selected_text}")
    except Exception:
        pass

    # Resolve the future so the polling loop can continue
    if _build_menu_future and not _build_menu_future.done():
        _build_menu_future.set_result(choice)


async def cmd_think(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Route query to a powerful cloud model via OpenRouter."""
    if not is_allowed(update.effective_user.id):
        return
    rl = _rate_check(update.effective_user.id, "cloud")
    if rl:
        await update.message.reply_text(rl)
        return

    args = update.message.text.split(maxsplit=1)
    if len(args) < 2:
        await update.message.reply_text("Usage: `/think <your question>`", parse_mode="Markdown")
        return

    query = args[1].strip()
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    cloud_model = get_user_cloud_model(chat_id)

    placeholder = await update.message.reply_text(f"🧠 Thinking with `{cloud_model}`...", parse_mode="Markdown")

    # Smart recall — fetch only relevant memories (per-user, per-agent)
    relevant_memory = await smart_recall(query, user_id, chat_id)

    # Build system prompt with relevant memory injected (agent-scoped)
    system = build_system_prompt_cloud(user_id, chat_id)
    if relevant_memory:
        system += f"\n\n---\n\n# Relevant Memories\n\n{relevant_memory}"

    # Build messages with history
    history = get_history(chat_id)
    messages = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({"role": "user", "content": query})

    response = await openrouter_chat(messages, model=cloud_model)

    # Offline fallback — if cloud failed, route through local Ollama
    fell_back = False
    if response.startswith("Cloud model error:") or response.startswith("OpenRouter API key not configured"):
        local_model = get_user_model(chat_id)
        try:
            await placeholder.edit_text(
                f"☁️ Cloud offline — falling back to local `{local_model}`...",
                parse_mode="Markdown",
            )
            local_messages = [{"role": "system", "content": SYSTEM_PROMPT_LOCAL}]
            local_messages.extend(history)
            local_messages.append({"role": "user", "content": query})
            response = ""
            async for token in ollama_stream(local_model, local_messages):
                response += token
            fell_back = True
        except Exception as e:
            log.error("Offline fallback also failed: %s", e)
            response = "❌ Both cloud and local models are unavailable right now. Try again in a bit."

    # Save to history
    save_message(chat_id, "user", query)
    save_message(chat_id, "assistant", response)

    # Truncate if too long for Telegram (4096 char limit)
    full_response = response
    if len(response) > 4000:
        response = response[:4000] + "\n\n... (truncated)"

    header = "📡 *Local fallback:*" if fell_back else "🧠 *Cloud response:*"
    try:
        await placeholder.edit_text(f"{header}\n\n{response}", parse_mode="Markdown")
    except Exception:
        await placeholder.edit_text(f"{header}\n\n{response}")

    # Background: auto-capture facts + profile update (per-user, per-agent)
    if AUTO_CAPTURE:
        asyncio.create_task(auto_capture(query, full_response, user_id, chat_id))

    global _think_counter
    _think_counter += 1
    if _think_counter % PROFILE_FREQUENCY == 0:
        asyncio.create_task(update_profile(chat_id, user_id))


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
    if not OPENROUTER_KEY:
        await update.message.reply_text(_friendly_error("API key not configured"))
        return
    rl = _rate_check(update.effective_user.id, "cloud")
    if rl:
        await update.message.reply_text(rl)
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
        await placeholder.edit_text(_friendly_error(data['error']))
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
        await placeholder.edit_text(_friendly_error(e, "Image generation"))


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
    if not OPENROUTER_KEY:
        await update.message.reply_text(_friendly_error("API key not configured"))
        return
    rl = _rate_check(update.effective_user.id, "cloud")
    if rl:
        await update.message.reply_text(rl)
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
        await placeholder.edit_text(_friendly_error(data['error']))
        return

    try:
        response = data["choices"][0]["message"]["content"]
        if len(response) > 4000:
            response = response[:4000] + "\n\n... (truncated)"
        await placeholder.edit_text(response)
    except Exception as e:
        log.error("See error: %s", e)
        await placeholder.edit_text(_friendly_error(e, "Vision"))


async def cmd_search(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/search <query> — Search the web and get a grounded answer."""
    if not is_allowed(update.effective_user.id):
        return
    if not OPENROUTER_KEY:
        await update.message.reply_text(_friendly_error("API key not configured"))
        return
    rl = _rate_check(update.effective_user.id, "cloud")
    if rl:
        await update.message.reply_text(rl)
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
        await placeholder.edit_text(_friendly_error(data['error']))
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
        await placeholder.edit_text(_friendly_error(e, "Search"))


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
    if not OPENROUTER_KEY:
        await update.message.reply_text(_friendly_error("API key not configured"))
        return
    rl = _rate_check(update.effective_user.id, "cloud")
    if rl:
        await update.message.reply_text(rl)
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
        await placeholder.edit_text(_friendly_error(data['error']))
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
        await placeholder.edit_text(_friendly_error(e, "Document"))


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
        # No voice note — show current voice + picker buttons
        chat_id = update.effective_chat.id
        current = get_user_voice(chat_id)
        current_name = next((k for k, (v, _) in TTS_VOICES.items() if v == current), "asteria")

        buttons = []
        row = []
        for name, (voice_id, desc) in TTS_VOICES.items():
            label = f"{'✅ ' if voice_id == current else ''}{name}"
            row.append(InlineKeyboardButton(label, callback_data=f"voice:{name}"))
            if len(row) == 3:
                buttons.append(row)
                row = []
        if row:
            buttons.append(row)

        await update.message.reply_text(
            f"🎙 *Voice Settings*\n\n"
            f"Current: *{current_name}* ({dict(TTS_VOICES.values()).get(current, '')})\n\n"
            f"Send a voice note and I'll respond with voice!\n"
            f"Tap below to change voice:",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(buttons),
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
        await placeholder.edit_text(_friendly_error(data['error']))
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
        await placeholder.edit_text(_friendly_error(e, "Voice"))


def _tmux_capture() -> str:
    """Capture the current tmux pane content."""
    import subprocess
    r = subprocess.run(
        ["tmux", "capture-pane", "-t", TMUX_SESSION, "-p"],
        capture_output=True, text=True,
    )
    return r.stdout if r.returncode == 0 else ""


def _extract_claude_response(pane_before: str, pane_now: str) -> str:
    """Extract new Claude Code output by comparing pane snapshots.

    Looks for lines starting with ● (Claude response marker) and other
    output lines that appeared after the message was sent.
    """
    before_lines = set(pane_before.strip().splitlines())
    now_lines = pane_now.strip().splitlines()

    new_lines = []
    collecting = False
    for line in now_lines:
        stripped = line.strip()
        # Skip empty lines at the start
        if not collecting and not stripped:
            continue
        # Start collecting from the ● marker (Claude's response)
        if stripped.startswith("●") or stripped.startswith("·"):
            collecting = True
        # Also collect continuation lines (indented text after ●)
        if collecting:
            # Stop at the input prompt line
            if stripped.startswith("❯") or stripped.startswith("⏵⏵"):
                break
            if "────────" in stripped:
                break
            new_lines.append(stripped)

    # Clean up: remove ● and · markers, join
    cleaned = []
    for line in new_lines:
        if line.startswith("●"):
            line = line[1:].strip()
        elif line.startswith("·"):
            line = line[1:].strip()
        elif line.startswith("⎿"):
            line = "  " + line[1:].strip()
        cleaned.append(line)

    return "\n".join(cleaned).strip()


def _detect_interactive_menu(pane: str) -> bool:
    """Detect if Claude Code is showing an interactive selection menu.

    Interactive menus (AskUserQuestion) show ❯ with text (selected option).
    The normal input prompt shows ❯ alone (empty). We check the bottom of
    the pane for ❯ followed by option text — that means a menu is blocking.
    """
    lines = pane.strip().splitlines()
    if not lines:
        return False
    # Check last 8 lines (where interactive UI appears)
    for line in lines[-8:]:
        s = line.strip()
        if s.startswith("❯ ") and len(s) > 3:
            return True
    return False


def _parse_interactive_menu(pane: str) -> tuple[str, list[str]]:
    """Extract question text and option labels from a Claude Code menu."""
    lines = pane.strip().splitlines()
    bottom = lines[-15:]

    options: list[str] = []
    selected_idx: int | None = None

    # Find the selected option (❯ marker)
    for i, line in enumerate(bottom):
        s = line.strip()
        if s.startswith("❯ ") and len(s) > 3:
            selected_idx = i
            options.append(s[2:])
            break

    if selected_idx is None:
        return "", []

    # Collect remaining options after the selected one
    for line in bottom[selected_idx + 1:]:
        s = line.strip()
        if not s:
            continue
        # Stop at markers that aren't options
        if s.startswith(("●", "·", "⎿", "⏵", "❯", "────")):
            break
        options.append(s)

    # Look for question text above the options
    question = ""
    for i in range(selected_idx - 1, max(selected_idx - 6, -1), -1):
        s = bottom[i].strip()
        if s and not s.startswith(("●", "·", "⎿", "⏵", "❯", "────")):
            question = s
            break

    return question, options


# Future for coordinating menu selection between polling loop and callback
_build_menu_future: asyncio.Future | None = None


async def _handle_build_message(update: Update, user_text: str, model: str):
    """Inject message into Claude Code tmux session and stream output back."""
    global _build_polling_cancel, _build_menu_future
    chat_id = update.effective_chat.id

    # If a previous polling loop is running, cancel it and interrupt Claude
    if _build_polling_cancel is not None and not _build_polling_cancel.is_set():
        _build_polling_cancel.set()
        _tmux_send_escape()
        await asyncio.sleep(1.5)

    # Check tmux session is alive
    if not _tmux_exists():
        _start_claude_tmux(model)
        await asyncio.sleep(3)
        if not _tmux_exists():
            exit_build_mode(chat_id)
            await update.message.reply_text(
                "❌ Claude Code session failed to start.\n\n"
                "Type `/build` to try again.",
                parse_mode="Markdown",
            )
            return

    # Write chat_id so the stop hook knows where to respond
    CHAT_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHAT_ID_FILE.write_text(str(chat_id))

    # Create a cancel event for this polling loop
    cancel = asyncio.Event()
    _build_polling_cancel = cancel

    # Snapshot pane BEFORE sending the message
    pane_before = _tmux_capture()

    # React to the message with a checkmark
    try:
        await update.message.set_reaction("✅")
    except Exception:
        pass

    # Inject the message into Claude Code via tmux
    # Small delay after long pastes so Claude Code's input component
    # finishes processing before Enter arrives
    _tmux_send(user_text)
    await asyncio.sleep(0.5)
    _tmux_send_enter()

    # Stream tmux output to Telegram by polling the pane
    bot = update.get_bot()
    msg = None  # The Telegram message we'll keep editing
    last_text = ""
    idle_count = 0
    max_wait = 300  # 5 minutes max

    for tick in range(max_wait):
        await asyncio.sleep(2)

        # Check if cancelled by a newer message
        if cancel.is_set():
            # Cancel any pending menu selection
            if _build_menu_future and not _build_menu_future.done():
                _build_menu_future.cancel()
            if msg and last_text:
                try:
                    await bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=msg.message_id,
                        text=last_text + "\n\n⚡ _Interrupted_",
                        parse_mode="Markdown",
                    )
                except Exception:
                    pass
            return

        # Check if session died
        if not _tmux_exists():
            if msg:
                try:
                    await bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=msg.message_id,
                        text=last_text + "\n\n⚠️ _Session ended_",
                        parse_mode="Markdown",
                    )
                except Exception:
                    pass
            break

        pane_now = _tmux_capture()
        response = _extract_claude_response(pane_before, pane_now)

        if not response:
            # Send typing while waiting for first output
            try:
                await bot.send_chat_action(chat_id, "typing")
            except Exception:
                pass
            continue

        # Check if output changed
        if response == last_text:
            idle_count += 1
            # If Claude's output hasn't changed for 8 seconds (4 ticks) and
            # the prompt input line is visible → Claude is done
            if idle_count >= 4:
                pane_check = _tmux_capture()
                lines = pane_check.strip().splitlines()
                # Check if the input prompt (❯) is back at the bottom
                for line in reversed(lines[-5:]):
                    s = line.strip()
                    if s.startswith("❯") and s == "❯":
                        idle_count = 999
                        break
                    if s.startswith("⏵⏵"):
                        idle_count = 999
                        break
                # Interactive menu detected — show options as Telegram buttons
                if idle_count < 999 and _detect_interactive_menu(pane_check):
                    question, options = _parse_interactive_menu(pane_check)
                    if len(options) > 1:
                        # Present choices to the user
                        loop = asyncio.get_running_loop()
                        menu_future = loop.create_future()
                        _build_menu_future = menu_future

                        buttons = []
                        for i, opt in enumerate(options):
                            label = (opt[:57] + "…") if len(opt) > 58 else opt
                            buttons.append([InlineKeyboardButton(
                                label, callback_data=f"bmenu:{i}",
                            )])

                        q_text = f"🔀 *Claude is asking:*\n{question}" if question else "🔀 *Claude needs your input:*"
                        await bot.send_message(
                            chat_id=chat_id,
                            text=q_text,
                            parse_mode="Markdown",
                            reply_markup=InlineKeyboardMarkup(buttons),
                        )

                        # Wait for user to tap a button (60s timeout)
                        try:
                            choice_idx = await asyncio.wait_for(menu_future, timeout=60)
                            # Navigate: Down arrow choice_idx times, then Enter
                            for _ in range(choice_idx):
                                _tmux_send("Down", literal=False)
                                await asyncio.sleep(0.15)
                            _tmux_send_enter()
                        except (asyncio.TimeoutError, asyncio.CancelledError):
                            _tmux_send_enter()  # Timeout → pick first option
                        finally:
                            _build_menu_future = None
                    else:
                        # Single option or can't parse — just Enter
                        _tmux_send_enter()
                    idle_count = 0
                if idle_count >= 999:
                    break
            continue
        else:
            idle_count = 0

        last_text = response

        # Truncate for Telegram (4096 char limit)
        display = response
        if len(display) > 4000:
            display = display[-3997:] + "..."

        try:
            if msg is None:
                msg = await bot.send_message(
                    chat_id=chat_id,
                    text=display,
                    disable_notification=True,
                )
            else:
                await bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=msg.message_id,
                    text=display,
                )
        except Exception:
            pass  # Edit failed (same content, rate limit, etc.)

    # Clear cancel ref if we're still the active loop
    if _build_polling_cancel is cancel:
        _build_polling_cancel = None


async def handle_build_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Handle photos sent in Build Mode — download and pass path to Claude Code."""
    log.info("handle_build_photo triggered")
    if not update.message or not is_allowed(update.effective_user.id):
        log.info("handle_build_photo: no message or not allowed")
        return
    chat_id = update.effective_chat.id
    build_model = get_build_mode(chat_id)
    log.info("handle_build_photo: chat_id=%s build_model=%s", chat_id, build_model)
    if not build_model:
        return  # Not in build mode, let group 1 handlers deal with it

    try:
        # Download the photo (largest size)
        photo = update.message.photo[-1]
        file = await photo.get_file()
        uploads = Path.home() / "projects" / "uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        filename = f"photo_{int(time.time())}_{photo.file_unique_id}.jpg"
        filepath = uploads / filename
        await file.download_to_drive(str(filepath))
        log.info("handle_build_photo: downloaded to %s", filepath)

        caption = update.message.caption or ""

        # Auto-describe the image with a vision model so Claude Code gets
        # an accurate description (Claude Code + Ollama can't reliably pass
        # images through the tool pipeline)
        description = ""
        try:
            import base64 as _b64
            img_b64 = _b64.b64encode(filepath.read_bytes()).decode()
            desc_prompt = (
                "Describe this image in detail for a developer who needs to "
                "recreate it. Include: layout, colors, typography, sections, "
                "text content, spacing, and visual style. Be specific."
            )
            # Direct call with a reliable vision model (not openrouter_raw
            # which raises on non-200 and loses the error body)
            async with httpx.AsyncClient(timeout=120.0) as _vc:
                _vr = await _vc.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "google/gemini-2.5-flash-lite-preview-09-2025",
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                                {"type": "text", "text": desc_prompt},
                            ],
                        }],
                    },
                )
                if _vr.status_code != 200:
                    log.error("Vision API %s: %s", _vr.status_code, _vr.text[:500])
                else:
                    desc_data = _vr.json()
                    description = desc_data["choices"][0]["message"]["content"]
                    log.info("handle_build_photo: vision OK (%d chars)", len(description))
        except Exception:
            log.exception("handle_build_photo: vision description failed")

        log.info("handle_build_photo: description=%s", "YES" if description else "EMPTY")

        # Build the message with the image path + vision description
        parts = [f"I sent you an image at {filepath}."]
        if description:
            parts.append(f"\n\nHere is a detailed description of what the image shows:\n{description}")
        if caption:
            parts.append(f"\n\nUser's instructions: {caption}")
        else:
            parts.append("\n\nPlease recreate what you see in this image.")
        msg = "".join(parts)

        asyncio.create_task(_handle_build_message(update, msg, build_model))
    except Exception:
        log.exception("handle_build_photo failed")
        await update.message.reply_text("❌ Failed to process photo. Check logs.")
    raise ApplicationHandlerStop  # Don't let /see handler also fire


async def handle_build_document(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Handle documents sent in Build Mode — download and pass path to Claude Code."""
    if not update.message or not update.message.document:
        return
    if not is_allowed(update.effective_user.id):
        return
    chat_id = update.effective_chat.id
    build_model = get_build_mode(chat_id)
    if not build_model:
        return

    doc = update.message.document
    file = await doc.get_file()
    uploads = Path.home() / "projects" / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    filename = doc.file_name or f"file_{int(time.time())}"
    filepath = uploads / filename
    await file.download_to_drive(str(filepath))

    caption = update.message.caption or ""
    if caption:
        msg = f"I sent you a file at {filepath} ({doc.file_name}). {caption}"
    else:
        msg = f"I sent you a file at {filepath} ({doc.file_name}). Please read it."

    asyncio.create_task(_handle_build_message(update, msg, build_model))
    raise ApplicationHandlerStop


async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Handle regular text messages — route to local Ollama or cloud (build mode)."""
    if not update.message or not update.message.text:
        return
    if not is_allowed(update.effective_user.id):
        return

    global _last_used_model, INTEGRATIONS
    chat_id = update.effective_chat.id
    user_text = update.message.text

    # Group chat: respond to all messages in routed groups (Zote, Oracle, etc.)
    # For non-routed groups: only respond if @mentioned or replied to
    if _is_group_chat(update):
        is_routed_group = str(chat_id) in GROUP_ROUTING
        if not is_routed_group:
            if not _bot_mentioned(update):
                return
        # Strip @mention from text if present
        if _bot_mentioned(update):
            user_text = _strip_mention(user_text)
            if not user_text:
                return

    # Check if we're waiting for an API key paste
    if chat_id in _api_awaiting_key and is_admin(update.effective_user.id):
        key_name = _api_awaiting_key.pop(chat_id)
        value = user_text.strip()
        # Save the key
        SECRETS[key_name] = value
        with open(SECRETS_PATH, "w") as f:
            json.dump(SECRETS, f, indent=2)
            f.write("\n")
        # Reload integrations
        INTEGRATIONS = load_integrations()
        # Delete the message (contains the API key)
        try:
            await update.message.delete()
        except Exception:
            pass
        masked = value[:8] + "..." if len(value) > 8 else "***"
        active = list(INTEGRATIONS.keys())
        await update.effective_chat.send_message(
            f"✅ `{key_name}` saved (`{masked}`)\n\n"
            f"Active integrations: {', '.join('/' + c for c in active) if active else 'none'}",
            parse_mode="Markdown",
        )
        return

    # Check if we're waiting for a Google auth code
    if chat_id in _gauth_awaiting_code:
        _gauth_awaiting_code.pop(chat_id)
        code = user_text.strip()
        if len(code) < 10:
            await update.message.reply_text("That doesn't look like a valid code. Try `/sheets` again.", parse_mode="Markdown")
            return
        # Route through the SAME module instance the integration system uses
        integration = INTEGRATIONS.get("gauth") or INTEGRATIONS.get("sheets")
        if not integration:
            await update.message.reply_text("Google integration not loaded.")
            return
        placeholder = await update.message.reply_text("Exchanging code...")
        try:
            result = await integration["handle"](
                command="gauth", args=code,
                secrets={k: get_secret(k) for k in SECRETS} if SECRETS else CFG,
                cloud_chat=lambda msgs, model=None: openrouter_chat(msgs, model=model),
            )
            if isinstance(result, dict):
                text = result.get("text", str(result))
                markup = result.get("reply_markup")
                try:
                    await placeholder.edit_text(text, parse_mode="Markdown", reply_markup=markup)
                except Exception:
                    await placeholder.edit_text(text, reply_markup=markup)
            else:
                try:
                    await placeholder.edit_text(str(result), parse_mode="Markdown")
                except Exception:
                    await placeholder.edit_text(str(result))
        except Exception as e:
            log.error("Google auth exchange error: %s", e)
            await placeholder.edit_text(f"Auth failed: {e}")
        return

    # Check if leads integration has an active session awaiting text input
    leads_int = INTEGRATIONS.get("leads")
    if leads_int:
        mod = leads_int.get("module")
        if mod and hasattr(mod, "has_active_session") and mod.has_active_session(chat_id):
            placeholder = await update.message.reply_text("Processing...")
            try:
                secrets = {k: get_secret(k) for k in SECRETS} if SECRETS else CFG
                result = await mod.handle_text(chat_id, user_text, secrets)
                if result:
                    if isinstance(result, dict):
                        text = result.get("text", "(no result)")
                        markup = result.get("reply_markup")
                        try:
                            await placeholder.edit_text(text, parse_mode="Markdown", reply_markup=markup)
                        except Exception:
                            await placeholder.edit_text(text, reply_markup=markup)
                    else:
                        if len(result) > 4000:
                            result = result[:4000] + "\n\n... (truncated)"
                        try:
                            await placeholder.edit_text(str(result), parse_mode="Markdown")
                        except Exception:
                            await placeholder.edit_text(str(result))
                else:
                    await placeholder.delete()
            except Exception as e:
                log.error("Leads text handler error: %s", e)
                await placeholder.edit_text(f"Error: {e}")
            return

    # Check if user is in Build Mode → route to actual Claude Code
    build_model = get_build_mode(chat_id)
    if build_model:
        # Run as background task so /exit and /stop commands aren't blocked
        asyncio.create_task(_handle_build_message(update, user_text, build_model))
        return

    # Rate limit — local chat
    rl = _rate_check(update.effective_user.id, "chat")
    if rl:
        await update.message.reply_text(rl)
        return

    model = get_user_model(chat_id)
    _last_used_model = model

    # Save user message
    save_message(chat_id, "user", user_text)

    # Cloud-only mode — route all messages through OpenRouter
    if CLOUD_ONLY:
        placeholder = await update.message.reply_text("...")
        user_id = update.effective_user.id
        cloud_model = get_user_cloud_model(chat_id)

        # Smart model routing: use sonnet for action intent, minimax for chat
        _action_kw = [
            "busca ", "encuentra ", "scrape", "scrapea",
            "quiero leads", "necesito leads", "dame leads",
            "buscar leads", "buscar ", "encontrar ",
            "find me ", "get me ", "search for ",
            "hazme un scrape", "genera leads", "consigue leads",
            "rastre", "investiga ",
        ]
        msg_lower = user_text.lower()
        if any(kw in msg_lower for kw in _action_kw):
            cloud_model = "anthropic/claude-sonnet-4-6"
            log.info("Action intent detected — routing to sonnet")

        system = build_system_prompt_cloud(user_id, chat_id)
        relevant_memory = ""
        try:
            relevant_memory = await smart_recall(user_text, user_id, chat_id)
        except Exception:
            pass
        if relevant_memory:
            system += f"\n\n---\n\n# Relevant Memories\n\n{relevant_memory}"
        history = get_history(chat_id)
        messages = [{"role": "system", "content": system}]
        messages.extend(history)
        try:
            response = await openrouter_chat(messages, model=cloud_model)
        except Exception as e:
            await placeholder.edit_text(f"Cloud error: {e}")
            return
        if not response.strip():
            response = "(empty response)"

        # Check for <<RUN:/command "args">> action tags from the model
        run_match = re.search(r'<<RUN:(/\w+)\s*(.*?)>>', response)
        if run_match:
            run_cmd = run_match.group(1).lstrip("/")
            run_args = run_match.group(2).strip().strip('"')
            # Strip the tag from the displayed response
            clean_response = response[:run_match.start()].strip()
            integration = INTEGRATIONS.get(run_cmd)
            if integration:
                if clean_response:
                    try:
                        await placeholder.edit_text(clean_response, parse_mode="Markdown")
                    except Exception:
                        await placeholder.edit_text(clean_response)
                else:
                    await placeholder.edit_text(f"Running /{run_cmd}...")

                secrets = {k: get_secret(k) for k in SECRETS} if SECRETS else CFG

                async def _cloud_chat(msgs, model=None):
                    return await openrouter_chat(msgs, model=model)

                try:
                    result = await integration["handle"](
                        command=run_cmd, args=run_args,
                        secrets=secrets, cloud_chat=_cloud_chat,
                    )
                    if isinstance(result, dict):
                        text = result.get("text", "")
                        markup = result.get("reply_markup")
                        full_text = f"{clean_response}\n\n{text}" if clean_response else text
                        if len(full_text) > 4000:
                            full_text = full_text[:4000] + "\n\n... (truncated)"
                        try:
                            await placeholder.edit_text(full_text, parse_mode="Markdown", reply_markup=markup)
                        except Exception:
                            await placeholder.edit_text(full_text, reply_markup=markup)
                    else:
                        result_str = str(result)
                        full_text = f"{clean_response}\n\n{result_str}" if clean_response else result_str
                        if len(full_text) > 4000:
                            full_text = full_text[:4000] + "\n\n... (truncated)"
                        try:
                            await placeholder.edit_text(full_text, parse_mode="Markdown")
                        except Exception:
                            await placeholder.edit_text(full_text)
                except Exception as e:
                    log.error("RUN action /%s failed: %s", run_cmd, e)
                    await placeholder.edit_text(f"{clean_response}\n\nError: {e}")
                save_message(chat_id, "assistant", clean_response or f"[Ran /{run_cmd}]")
                return

        if len(response) > 4000:
            response = response[:4000] + "\n\n... (truncated)"
        try:
            await placeholder.edit_text(response, parse_mode="Markdown")
        except Exception:
            try:
                await placeholder.edit_text(response)
            except Exception:
                pass
        save_message(chat_id, "assistant", response)
        if AUTO_CAPTURE:
            asyncio.create_task(auto_capture(user_text, response, user_id, chat_id))
        return

    # Normal mode — local prompt with optional memory recall
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
            relevant_memory = await smart_recall(user_text, update.effective_user.id, chat_id)
            if relevant_memory:
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
        await placeholder.edit_text(_friendly_error(e, "Chat"))
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
    # Group chat: only respond to voice if it's a reply to the bot
    if _is_group_chat(update) and not _bot_mentioned(update):
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
        user_id = update.effective_user.id
        cloud_model = get_user_cloud_model(chat_id)
        relevant_memory = await smart_recall(transcript, user_id, chat_id)
        system = build_system_prompt_cloud(user_id, chat_id)
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
        user_voice = get_user_voice(chat_id)
        audio_response = await synthesize(tts_text, deepgram_key, voice=user_voice)

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

        # Background: auto-capture (per-user, per-agent)
        if AUTO_CAPTURE:
            asyncio.create_task(auto_capture(transcript, response, user_id, chat_id))

    except Exception as e:
        log.error("Voice handler error: %s", e)
        await placeholder.edit_text(_friendly_error(e, "Voice"))


# ---------------------------------------------------------------------------
# Admin Dashboard — lightweight web status page
# ---------------------------------------------------------------------------

DASHBOARD_PORT = CFG.get("dashboard_port", 7777)
DASHBOARD_PASSWORD = CFG.get("dashboard_password", "berryclaw")
_bot_start_time: float = time.time()


def _dashboard_stats() -> dict:
    """Gather stats for the dashboard."""
    import subprocess

    now = time.time()
    uptime_s = int(now - _bot_start_time)
    days, rem = divmod(uptime_s, 86400)
    hours, rem = divmod(rem, 3600)
    mins, _ = divmod(rem, 60)
    bot_uptime = f"{days}d {hours}h {mins}m" if days else f"{hours}h {mins}m"

    try:
        sys_uptime = subprocess.check_output("uptime -p", shell=True, text=True).strip()
    except Exception:
        sys_uptime = "unknown"

    try:
        mem_info = subprocess.check_output("free -m", shell=True, text=True)
        parts = mem_info.split("\n")[1].split()
        mem_total, mem_used = int(parts[1]), int(parts[2])
        mem_pct = round(mem_used / mem_total * 100)
    except Exception:
        mem_total, mem_used, mem_pct = 0, 0, 0

    try:
        temp_raw = subprocess.check_output(
            "cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo 0",
            shell=True, text=True,
        ).strip()
        cpu_temp = f"{int(temp_raw) / 1000:.1f}"
    except Exception:
        cpu_temp = "N/A"

    try:
        disk = subprocess.check_output(
            "df -h / | awk 'NR==2{print $3\"/\"$2\" (\"$5\")\"}'",
            shell=True, text=True,
        ).strip()
    except Exception:
        disk = "unknown"

    # DB stats
    try:
        total_msgs = DB.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        unique_users = DB.execute("SELECT COUNT(DISTINCT chat_id) FROM messages").fetchone()[0]
        msgs_today = DB.execute(
            "SELECT COUNT(*) FROM messages WHERE ts > ?", (now - 86400,)
        ).fetchone()[0]
        recent_users = DB.execute(
            "SELECT DISTINCT chat_id FROM messages WHERE ts > ? ORDER BY ts DESC LIMIT 20",
            (now - 86400,),
        ).fetchall()
        recent_user_ids = [r[0] for r in recent_users]
    except Exception:
        total_msgs, unique_users, msgs_today = 0, 0, 0
        recent_user_ids = []

    # Active build sessions
    try:
        build_sessions = DB.execute("SELECT COUNT(*) FROM build_mode").fetchone()[0]
    except Exception:
        build_sessions = 0

    return {
        "bot_uptime": bot_uptime,
        "sys_uptime": sys_uptime,
        "mem_used": mem_used,
        "mem_total": mem_total,
        "mem_pct": mem_pct,
        "cpu_temp": cpu_temp,
        "disk": disk,
        "total_msgs": total_msgs,
        "unique_users": unique_users,
        "msgs_today": msgs_today,
        "recent_user_ids": recent_user_ids,
        "build_sessions": build_sessions,
        "default_model": DEFAULT_MODEL,
        "cloud_model": OPENROUTER_MODEL,
        "ollama_url": OLLAMA_URL,
        "admin_count": len(ADMIN_USERS),
        "rate_chat": RATE_LIMITS["chat"],
        "rate_cloud": RATE_LIMITS["cloud"],
    }


_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta http-equiv="refresh" content="30">
<title>Berryclaw Dashboard</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:#f8fafc;color:#1e293b;padding:20px;max-width:900px;margin:0 auto}
h1{font-size:1.5rem;margin-bottom:20px;display:flex;align-items:center;gap:8px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px;margin-bottom:24px}
.card{background:#fff;border-radius:10px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
.card .label{font-size:.75rem;color:#64748b;text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px}
.card .value{font-size:1.4rem;font-weight:600}
.card .sub{font-size:.8rem;color:#94a3b8;margin-top:2px}
.section{margin-bottom:24px}
.section h2{font-size:1.1rem;margin-bottom:10px;color:#475569}
table{width:100%;border-collapse:collapse;background:#fff;border-radius:10px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.08)}
th,td{text-align:left;padding:10px 14px;border-bottom:1px solid #f1f5f9}
th{background:#f8fafc;font-size:.75rem;color:#64748b;text-transform:uppercase;letter-spacing:.5px}
td{font-size:.9rem}
.bar{height:8px;border-radius:4px;background:#e2e8f0;overflow:hidden;margin-top:6px}
.bar-fill{height:100%;border-radius:4px;transition:width .3s}
.green{background:#22c55e}.yellow{background:#eab308}.red{background:#ef4444}
.badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:.75rem;font-weight:500}
.badge-ok{background:#dcfce7;color:#166534}.badge-warn{background:#fef9c3;color:#854d0e}
footer{text-align:center;color:#94a3b8;font-size:.8rem;margin-top:30px}
</style></head><body>
<h1>🫐 Berryclaw Dashboard</h1>
<div class="grid">
  <div class="card">
    <div class="label">Bot Uptime</div>
    <div class="value">{bot_uptime}</div>
    <div class="sub">System: {sys_uptime}</div>
  </div>
  <div class="card">
    <div class="label">RAM</div>
    <div class="value">{mem_used} / {mem_total} MB</div>
    <div class="bar"><div class="bar-fill {mem_color}" style="width:{mem_pct}%"></div></div>
  </div>
  <div class="card">
    <div class="label">CPU Temp</div>
    <div class="value">{cpu_temp}°C</div>
    <div class="sub">Disk: {disk}</div>
  </div>
  <div class="card">
    <div class="label">Messages Today</div>
    <div class="value">{msgs_today}</div>
    <div class="sub">{total_msgs} total</div>
  </div>
  <div class="card">
    <div class="label">Users</div>
    <div class="value">{unique_users}</div>
    <div class="sub">{active_today} active today</div>
  </div>
  <div class="card">
    <div class="label">Build Sessions</div>
    <div class="value">{build_sessions}</div>
    <div class="sub">Active now</div>
  </div>
</div>

<div class="section">
  <h2>Configuration</h2>
  <table>
    <tr><th>Setting</th><th>Value</th></tr>
    <tr><td>Local model</td><td><code>{default_model}</code></td></tr>
    <tr><td>Cloud model</td><td><code>{cloud_model}</code></td></tr>
    <tr><td>Ollama URL</td><td>{ollama_url}</td></tr>
    <tr><td>Admin users</td><td>{admin_count}</td></tr>
    <tr><td>Rate limit (chat)</td><td>{rate_chat_max}/min</td></tr>
    <tr><td>Rate limit (cloud)</td><td>{rate_cloud_max}/min</td></tr>
  </table>
</div>

{recent_section}

<footer>Auto-refreshes every 30 seconds · Berryclaw admin dashboard</footer>
</body></html>"""


def _render_dashboard() -> str:
    stats = _dashboard_stats()
    mem_color = "green" if stats["mem_pct"] < 60 else ("yellow" if stats["mem_pct"] < 85 else "red")

    recent_section = ""
    if stats["recent_user_ids"]:
        rows = "".join(f"<tr><td>{uid}</td></tr>" for uid in stats["recent_user_ids"])
        recent_section = (
            '<div class="section"><h2>Active Users (last 24h)</h2>'
            f'<table><tr><th>User ID</th></tr>{rows}</table></div>'
        )

    return _DASHBOARD_HTML.format(
        bot_uptime=stats["bot_uptime"],
        sys_uptime=stats["sys_uptime"],
        mem_used=stats["mem_used"],
        mem_total=stats["mem_total"],
        mem_pct=stats["mem_pct"],
        mem_color=mem_color,
        cpu_temp=stats["cpu_temp"],
        disk=stats["disk"],
        msgs_today=stats["msgs_today"],
        total_msgs=stats["total_msgs"],
        unique_users=stats["unique_users"],
        active_today=len(stats["recent_user_ids"]),
        build_sessions=stats["build_sessions"],
        default_model=stats["default_model"],
        cloud_model=stats["cloud_model"],
        ollama_url=stats["ollama_url"],
        admin_count=stats["admin_count"],
        rate_chat_max=stats["rate_chat"][0],
        rate_cloud_max=stats["rate_cloud"][0],
        recent_section=recent_section,
    )


def _start_dashboard():
    """Start the admin dashboard HTTP server in a background thread."""
    if DASHBOARD_PORT <= 0:
        log.info("Dashboard disabled (port <= 0)")
        return

    import base64
    import http.server
    import threading

    class DashboardHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # Silence access logs

        def _check_auth(self) -> bool:
            auth = self.headers.get("Authorization", "")
            if auth.startswith("Basic "):
                try:
                    decoded = base64.b64decode(auth[6:]).decode()
                    user, pw = decoded.split(":", 1)
                    if user == "admin" and pw == DASHBOARD_PASSWORD:
                        return True
                except Exception:
                    pass
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="Berryclaw Dashboard"')
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Login required")
            return False

        def do_GET(self):
            if not self._check_auth():
                return
            if self.path == "/api/stats":
                import json as _json
                stats = _dashboard_stats()
                stats["recent_user_ids"] = [str(u) for u in stats["recent_user_ids"]]
                stats["rate_chat"] = list(stats["rate_chat"])
                stats["rate_cloud"] = list(stats["rate_cloud"])
                body = _json.dumps(stats).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
                return
            html = _render_dashboard().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html)

    def run():
        try:
            server = http.server.HTTPServer(("0.0.0.0", DASHBOARD_PORT), DashboardHandler)
            log.info("Dashboard running on http://0.0.0.0:%d", DASHBOARD_PORT)
            server.serve_forever()
        except Exception as e:
            log.error("Dashboard failed to start: %s", e)

    t = threading.Thread(target=run, daemon=True)
    t.start()


async def _check_first_run_models(application):
    """On startup, if Ollama has 0 local models, notify admins with pull buttons."""
    if CLOUD_ONLY:
        return  # No Ollama needed in cloud-only mode
    await asyncio.sleep(3)  # Let bot fully start
    try:
        models = await ollama_list_models()
        local_models = [m for m in models if not _is_cloud_model(m)]
        if local_models:
            return  # Already has models — nothing to do

        # Check if Ollama is even reachable
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            r.raise_for_status()

        # Ollama is up but no local models — send admins a message
        buttons = []
        for name, desc, size in RECOMMENDED_LOCAL_MODELS:
            buttons.append([InlineKeyboardButton(
                f"📥 {name.split('/')[-1]} — {desc} ({size})",
                callback_data=f"pull:{name}",
            )])

        for admin_id in ADMIN_USERS:
            try:
                await application.bot.send_message(
                    chat_id=admin_id,
                    text=(
                        "📦 *No local models found on Ollama.*\n\n"
                        "Pick one to download — I'll set it up for you:"
                    ),
                    parse_mode="Markdown",
                    reply_markup=InlineKeyboardMarkup(buttons),
                )
            except Exception as e:
                log.warning("Could not notify admin %s about models: %s", admin_id, e)

        log.info("No local models found — notified %d admin(s)", len(ADMIN_USERS))
    except Exception as e:
        log.warning("First-run model check failed (Ollama down?): %s", e)


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

    # Start admin dashboard
    _start_dashboard()

    app = Application.builder().token(BOT_TOKEN).concurrent_updates(True).build()

    # Commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("model", cmd_model))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("build", cmd_build))
    app.add_handler(CommandHandler("exit", cmd_exit))
    app.add_handler(CommandHandler("stop", cmd_stop))
    app.add_handler(CommandHandler("claude", cmd_claude))
    app.add_handler(CommandHandler("auth", cmd_auth))
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
    app.add_handler(CallbackQueryHandler(callback_build, pattern=r"^build:"))
    app.add_handler(CallbackQueryHandler(callback_build_menu, pattern=r"^bmenu:"))
    app.add_handler(CallbackQueryHandler(callback_voice, pattern=r"^voice:"))
    app.add_handler(CallbackQueryHandler(callback_api, pattern=r"^api:"))
    app.add_handler(CallbackQueryHandler(callback_google, pattern=r"^g[sd]:"))
    app.add_handler(CallbackQueryHandler(callback_leads, pattern=r"^leads:"))
    app.add_handler(CallbackQueryHandler(
        lambda u, c: log.warning("Unhandled callback: %s", u.callback_query.data)
    ))

    # Build mode media — intercept photos/documents when in build mode
    app.add_handler(MessageHandler(filters.PHOTO, handle_build_photo), group=0)
    app.add_handler(MessageHandler(filters.Document.ALL, handle_build_document), group=0)

    # Photos with /see caption
    app.add_handler(MessageHandler(
        filters.PHOTO & filters.CaptionRegex(r"^/see"),
        cmd_see,
    ), group=1)
    # Documents with /read caption
    app.add_handler(MessageHandler(
        filters.Document.ALL & filters.CaptionRegex(r"^/read"),
        cmd_read,
    ), group=1)

    # Voice messages — auto voice-in/voice-out with Deepgram
    app.add_handler(MessageHandler(filters.VOICE, handle_voice_message))

    # Regular messages
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    async def post_init(application):
        global _bot_app, _bot_username
        _bot_app = application
        me = await application.bot.get_me()
        _bot_username = (me.username or "").lower()
        log.info("Bot username: @%s", _bot_username)

        # Register /commands in Telegram's menu
        from telegram import BotCommand
        await application.bot.set_my_commands([
            BotCommand("start", "Setup guide & health check"),
            BotCommand("help", "List all commands"),
            BotCommand("think", "Ask a powerful cloud model"),
            BotCommand("build", "Start Claude Code session"),
            BotCommand("exit", "Exit Claude Code"),
            BotCommand("stop", "Interrupt Claude Code"),
            BotCommand("claude", "View/add Build Mode rules"),
            BotCommand("auth", "View/change project login"),
            BotCommand("imagine", "Generate an image"),
            BotCommand("see", "Analyze a photo"),
            BotCommand("search", "Search the web"),
            BotCommand("read", "Read & summarize a document"),
            BotCommand("voice", "Pick TTS voice"),
            BotCommand("model", "Switch local model"),
            BotCommand("modelx", "Switch cloud model"),
            BotCommand("soul", "Change personality"),
            BotCommand("skills", "List available skills"),
            BotCommand("remember", "Save a memory"),
            BotCommand("memory", "View all memories"),
            BotCommand("profile", "View your user profile"),
            BotCommand("api", "Manage API keys"),
            BotCommand("status", "Pi system stats"),
            BotCommand("clear", "Reset conversation"),
        ])

        asyncio.create_task(warmup_loop())
        asyncio.create_task(heartbeat_loop())

        # Auto-pull check: if Ollama has 0 models, notify admins
        asyncio.create_task(_check_first_run_models(application))

    app.post_init = post_init

    log.info("Bot is running. Press Ctrl+C to stop.")
    app.run_polling(
        drop_pending_updates=True,
        allowed_updates=["message", "callback_query"],
    )


def setup_wizard():
    """Interactive setup wizard — creates config.json and secrets.json."""
    import sys

    print("\n🫐 Berryclaw Setup\n")
    print("This wizard will create your config.json and secrets.json files.\n")

    # --- secrets.json ---
    secrets = {}
    if SECRETS_PATH.exists():
        with open(SECRETS_PATH) as f:
            secrets = json.load(f)
        print(f"Found existing secrets.json — will update it.\n")

    # Telegram bot token
    current_token = secrets.get("telegram_bot_token", "")
    if current_token and current_token != "YOUR_BOT_TOKEN_FROM_BOTFATHER":
        print(f"Telegram bot token: {current_token[:8]}...{current_token[-4:]}")
        change = input("Keep this token? [Y/n] ").strip().lower()
        if change == "n":
            current_token = ""
    if not current_token or current_token == "YOUR_BOT_TOKEN_FROM_BOTFATHER":
        print("Get a bot token from @BotFather on Telegram: https://t.me/BotFather")
        current_token = input("Paste your bot token: ").strip()
        if not current_token:
            print("❌ Bot token is required. Exiting.")
            sys.exit(1)
    secrets["telegram_bot_token"] = current_token

    # OpenRouter API key
    current_or = secrets.get("openrouter_api_key", "")
    if current_or and current_or != "YOUR_OPENROUTER_API_KEY":
        print(f"\nOpenRouter API key: {current_or[:8]}...")
        change = input("Keep this key? [Y/n] ").strip().lower()
        if change == "n":
            current_or = ""
    if not current_or or current_or == "YOUR_OPENROUTER_API_KEY":
        print("\nOpenRouter powers /think, /imagine, /search, and more.")
        print("Get a free key at: https://openrouter.ai/keys")
        current_or = input("Paste your OpenRouter key (or Enter to skip): ").strip()
    secrets["openrouter_api_key"] = current_or

    # Ollama API key (for cloud models / Build Mode)
    current_ollama = secrets.get("ollama_api_key", "")
    if not current_ollama:
        print("\nOllama cloud key enables Build Mode (Claude Code with cloud models).")
        print("Sign in at: https://ollama.com then run 'ollama signin'")
        current_ollama = input("Paste your Ollama API key (or Enter to skip): ").strip()
    secrets["ollama_api_key"] = current_ollama

    # Ensure other keys exist
    for key in ["deepgram_api_key", "firecrawl_api_key", "apify_api_key", "google_credentials_file"]:
        if key not in secrets:
            secrets[key] = ""

    with open(SECRETS_PATH, "w") as f:
        json.dump(secrets, f, indent=2)
        f.write("\n")
    print(f"\n✅ Saved secrets.json")

    # --- config.json ---
    config = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            config = json.load(f)
        print(f"Found existing config.json — will update it.\n")

    # Admin user ID
    current_admins = config.get("admin_users", [])
    if current_admins:
        print(f"Admin users: {current_admins}")
        change = input("Keep these admins? [Y/n] ").strip().lower()
        if change == "n":
            current_admins = []
    if not current_admins:
        print("\nYour Telegram user ID makes you the bot admin.")
        print("To find it: message @userinfobot on Telegram")
        admin_input = input("Your Telegram user ID (or Enter to auto-detect on first /start): ").strip()
        if admin_input:
            try:
                current_admins = [int(admin_input)]
            except ValueError:
                print("⚠️ Invalid ID — will auto-detect on first /start")
                current_admins = []
    config["admin_users"] = current_admins

    # Ollama URL
    ollama_url = config.get("ollama_url", "http://localhost:11434")
    print(f"\nOllama URL: {ollama_url}")
    new_url = input("Change? (Enter to keep): ").strip()
    if new_url:
        ollama_url = new_url
    config["ollama_url"] = ollama_url

    # Default model
    default = config.get("default_model", "qwen25-pi")
    print(f"\nDefault local model: {default}")
    print("Recommended models for Raspberry Pi 5:")
    for i, (name, desc, size) in enumerate(RECOMMENDED_LOCAL_MODELS, 1):
        marker = " ← recommended" if i == 1 else ""
        print(f"  {i}. {name} — {desc} ({size}){marker}")
    new_model = input("Pick a number, type a model name, or Enter to keep current: ").strip()
    if new_model:
        if new_model.isdigit() and 1 <= int(new_model) <= len(RECOMMENDED_LOCAL_MODELS):
            default = RECOMMENDED_LOCAL_MODELS[int(new_model) - 1][0]
        else:
            default = new_model
    config["default_model"] = default

    # Fill in defaults for other settings
    config.setdefault("max_history", 10)
    config.setdefault("stream_batch_tokens", 15)
    config.setdefault("warmup_interval_seconds", 240)
    config.setdefault("heartbeat_interval_seconds", 1800)
    config.setdefault("allowed_users", [])
    config.setdefault("openrouter_model", "x-ai/grok-4.1-fast")
    config.setdefault("memory_model", "liquid/lfm-2.5-1.2b-instruct:free")
    config.setdefault("auto_capture", True)
    config.setdefault("profile_frequency", 20)

    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    print(f"✅ Saved config.json")

    print(f"\n🫐 Setup complete! Run the bot:\n")
    print(f"   python3 berryclaw.py\n")
    print(f"Then send /start to your bot on Telegram.")
    if not current_admins:
        print(f"The first person to /start will become admin.\n")


if __name__ == "__main__":
    import sys
    if "--setup" in sys.argv:
        setup_wizard()
    else:
        main()
