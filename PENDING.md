# Berryclaw — Pending Features for Public Release

## Priority 1 — Must Have

- [x] **Interactive setup** — `python3 berryclaw.py --setup` wizard + auto-admin on first `/start`. Install script auto-runs setup if no config exists.
- [x] **Multi-user memory** — Per-user MEMORY.md and PROFILE.md stored in `workspace/memory/<user_id>/`. Auto-migrates from legacy global files. All commands (/remember, /memory, /forget, /profile, /think, voice) are per-user.
- [x] **Offline fallback** — `/think` falls back to local Ollama when OpenRouter is down. Shows "Cloud offline — falling back to local" message.

## Priority 2 — Important

- [x] **Group chat support** — Respond only when @mentioned or replied to. Per-group conversation history. Group-aware personality.
- [x] **Voice output (TTS)** — Deepgram integration with `/voice` picker (12 voices, per-user selection).
- [x] **Model auto-pull** — On first run, if no Ollama models exist, notifies admin with 3 recommended models to pick from. Also shown in /start and /model.
- [x] **Docker option** — `docker-compose.yml` with Berryclaw + Ollama for one-command deploy. Includes Dockerfile, .dockerignore, OLLAMA_HOST env var support.

## Priority 3 — Polish

- [x] **LICENSE file** — MIT license file.
- [x] **CONTRIBUTING.md** — How to contribute, code style, PR process.
- [x] **GitHub issue templates** — Bug report, feature request templates.
- [x] **Better error messages** — `_friendly_error()` helper maps raw exceptions to user-friendly messages. API key guards on all cloud commands.
- [x] **Rate limiting** — Per-user rate limits (30 chat/min, 10 cloud/min). Admins exempt. Configurable via config.json.
- [ ] **Admin web dashboard** — Simple status page accessible from browser (Pi IP:port) showing stats, active users, memory usage, model info.
- [ ] **Scheduled messages / reminders** — "Remind me in 2 hours to check the oven" via local scheduling.
- [ ] **Inline mode** — Telegram inline queries for quick answers without opening the chat.
