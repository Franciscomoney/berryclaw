"""Berryclaw Integrations — auto-discovered API-powered skills.

Each integration is a Python file in this folder with:
  NAME: str                    — integration name
  COMMANDS: dict[str, str]     — {command_name: description}
  REQUIRED_SECRETS: list[str]  — secrets needed from secrets.json
  async def handle(command, args, secrets, cloud_chat) -> str
"""
