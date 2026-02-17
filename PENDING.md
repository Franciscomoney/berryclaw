# Berryclaw — Pending Features for Public Release

## Priority 1 — Must Have

- [ ] **Interactive setup** — `python3 berryclaw.py --setup` wizard that asks for bot token, OpenRouter key, preferred model, admin user ID. Writes config.json. Alternatively, `/setup` command in Telegram for first admin.
- [ ] **Multi-user memory** — MEMORY.md and PROFILE.md are currently global. Need per-user memory and profile (SQLite or separate files per user_id). Conversation history is already per chat_id.
- [ ] **Offline fallback** — If OpenRouter is down or no internet, gracefully route /think through Ollama instead of failing silently. Show "Cloud offline, using local brain" message.

## Priority 2 — Important

- [ ] **Group chat support** — Respond only when @mentioned or replied to. Per-group conversation history. Group-aware personality.
- [ ] **Voice output (TTS)** — Send voice note responses back. Use piper-tts locally or a free TTS model on OpenRouter.
- [ ] **Model auto-pull** — On first run, if no Ollama models exist, auto-pull a recommended model (e.g., `huihui_ai/qwen2.5-abliterate:1.5b`) or guide the user.
- [ ] **Docker option** — `docker-compose.yml` with Berryclaw + Ollama for one-command deploy.

## Priority 3 — Polish

- [ ] **LICENSE file** — MIT license file (README says MIT but no file exists).
- [ ] **CONTRIBUTING.md** — How to contribute, code style, PR process.
- [ ] **GitHub issue templates** — Bug report, feature request templates.
- [ ] **Better error messages** — User-friendly errors when Ollama isn't running, bad API key, model not found, etc.
- [ ] **Rate limiting** — Per-user rate limits to prevent abuse when bot is public.
- [ ] **Admin web dashboard** — Simple status page accessible from browser (Pi IP:port) showing stats, active users, memory usage, model info.
- [ ] **Scheduled messages / reminders** — "Remind me in 2 hours to check the oven" via local scheduling.
- [ ] **Inline mode** — Telegram inline queries for quick answers without opening the chat.
