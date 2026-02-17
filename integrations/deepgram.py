"""Deepgram integration — voice-in, voice-out conversations.

Send a voice note → transcribe → AI responds → TTS voice note back.
Also handles /voice command for manual transcription.
"""

import io

import httpx

NAME = "deepgram"
COMMANDS = {
    "voicechat": "Voice conversation — send voice, get voice back",
}
REQUIRED_SECRETS = ["deepgram_api_key"]

STT_URL = "https://api.deepgram.com/v1/listen"
TTS_URL = "https://api.deepgram.com/v1/speak"

# Deepgram TTS voices — aura model
DEFAULT_VOICE = "aura-asteria-en"  # Female, natural
VOICES = {
    "asteria": "aura-asteria-en",      # Female, warm
    "luna": "aura-luna-en",            # Female, soft
    "stella": "aura-stella-en",        # Female, clear
    "athena": "aura-athena-en",        # Female, professional
    "hera": "aura-hera-en",           # Female, authoritative
    "orion": "aura-orion-en",         # Male, deep
    "arcas": "aura-arcas-en",         # Male, conversational
    "perseus": "aura-perseus-en",     # Male, warm
    "angus": "aura-angus-en",         # Male, Irish
    "orpheus": "aura-orpheus-en",     # Male, rich
    "helios": "aura-helios-en",       # Male, British
    "zeus": "aura-zeus-en",           # Male, authoritative
}


async def transcribe(audio_bytes: bytes, api_key: str, mime: str = "audio/ogg") -> str:
    """Transcribe audio bytes to text using Deepgram STT."""
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": mime,
    }
    params = {
        "model": "nova-3",
        "smart_format": "true",
        "detect_language": "true",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(
            STT_URL,
            headers=headers,
            params=params,
            content=audio_bytes,
        )
        r.raise_for_status()
        data = r.json()

    # Extract transcript
    channels = data.get("results", {}).get("channels", [])
    if not channels:
        return ""

    alternatives = channels[0].get("alternatives", [])
    if not alternatives:
        return ""

    return alternatives[0].get("transcript", "")


async def synthesize(text: str, api_key: str, voice: str = DEFAULT_VOICE) -> bytes:
    """Convert text to speech using Deepgram TTS. Returns audio bytes (mp3)."""
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json",
    }
    params = {
        "model": voice,
        "encoding": "mp3",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(
            TTS_URL,
            headers=headers,
            params=params,
            json={"text": text},
        )
        r.raise_for_status()
        return r.content


async def handle(command: str, args: str, secrets: dict, cloud_chat) -> str:
    """Handle /voicechat command (text-based fallback)."""
    api_key = secrets.get("deepgram_api_key", "")

    if command == "voicechat":
        if not args:
            voice_list = "\n".join(f"  `{k}` — {v}" for k, v in list(VOICES.items())[:6])
            return (
                "*Voice Chat*\n\n"
                "Send a voice message and I'll respond with voice!\n\n"
                "Or type: `/voicechat <text>` to hear me say something.\n\n"
                "*Available voices:*\n" + voice_list + "\n\n"
                "Change voice: `/voicechat voice <name>`"
            )

        # /voicechat voice <name> — just info
        if args.startswith("voice"):
            voice_list = "\n".join(f"  `{k}` — {v}" for k, v in VOICES.items())
            return f"*Available voices:*\n{voice_list}"

        # /voicechat <text> — TTS only
        try:
            audio = await synthesize(args[:500], api_key)
            return f"__TTS_AUDIO__{len(audio)}"  # Marker for the handler
        except Exception as e:
            return f"TTS error: {e}"

    return "Unknown command."
