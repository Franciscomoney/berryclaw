# Contributing to Berryclaw

Thanks for your interest in Berryclaw! Here's how to contribute.

## Getting Started

1. Fork the repo and clone it
2. Run `./install.sh` or set up manually (see README)
3. Make your changes in a new branch

## Code Style

- **Single file** — all bot logic lives in `berryclaw.py`. Keep it that way.
- **No frameworks** — no Django, no Flask for the bot itself. Just `python-telegram-bot` + `httpx`.
- **Small prompts** — if you're adding a feature that touches the local model's system prompt, keep it short. Small models choke on long instructions.
- **Type hints** — use them for function signatures.
- **Logging** — use `log.info()` / `log.error()`, not `print()`.

## Adding Integrations

Integrations live in `integrations/` as standalone Python files. Each one:

1. Defines `INTEGRATION_META` with name, commands, and required API key
2. Registers handlers via `register(app, get_secret_fn)`
3. Is auto-loaded on startup — no changes to `berryclaw.py` needed

See existing integrations for the pattern.

## Pull Requests

1. One feature or fix per PR
2. Test on a real Pi if possible (or at least a Linux machine with Ollama)
3. Keep the PR description short — what changed and why
4. If it's a new command, update the README command table

## Reporting Bugs

Open an issue with:

- What you expected
- What happened instead
- Your setup (Pi model, OS, Python version, Ollama model)
- Relevant log output (`tail -50 raspberryclaw.log`)

## Feature Ideas

Open an issue first to discuss before building. Check PENDING.md for what's already planned.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
