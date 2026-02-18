"""Google Workspace integration — read/write Google Sheets and Docs.

Auth: OAuth2 flow via Telegram using InstalledAppFlow (same as OpenClaw).
  1. User types /sheets or /docs
  2. If no token, bot shows "Authorize" button
  3. User taps → bot generates auth URL via InstalledAppFlow, sends as button
  4. User authorizes in browser, gets code
  5. User pastes code → bot exchanges via InstalledAppFlow, saves token.pickle
"""

import json
import logging
import pickle
import time
from pathlib import Path

import httpx
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

log = logging.getLogger("berryclaw.google")

NAME = "google"
COMMANDS = {
    "sheets": "Read or write Google Sheets",
    "docs": "Read or edit Google Docs",
    "gauth": "Authorize Google account (OAuth2)",
}
REQUIRED_SECRETS = []

SHEETS_API = "https://sheets.googleapis.com/v4/spreadsheets"
DOCS_API = "https://docs.googleapis.com/v1/documents"
TOKEN_URL = "https://oauth2.googleapis.com/token"

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/drive",
]

_GWORK_DIR: Path | None = None
_cached_token: dict = {"token": "", "expires": 0}
# Store the flow object so we can exchange the code later
_pending_flow = {}  # chat_id → InstalledAppFlow


def _gwork_dir() -> Path:
    global _GWORK_DIR
    if _GWORK_DIR is None:
        _GWORK_DIR = Path(__file__).resolve().parent.parent / "google-workspace"
    return _GWORK_DIR


def _creds_path() -> Path:
    return _gwork_dir() / "credentials.json"


def _token_path() -> Path:
    return _gwork_dir() / "token.pickle"


def _load_token():
    tp = _token_path()
    if tp.exists():
        with open(tp, "rb") as f:
            return pickle.load(f)
    return None


def _save_token(creds) -> None:
    _gwork_dir().mkdir(parents=True, exist_ok=True)
    with open(_token_path(), "wb") as f:
        pickle.dump(creds, f)
    log.info(f"Token saved to {_token_path()}")


async def _get_access_token() -> str:
    global _cached_token
    now = time.time()
    if _cached_token["token"] and _cached_token["expires"] > now + 60:
        return _cached_token["token"]

    creds = _load_token()
    if creds is None:
        raise PermissionError("NOT_AUTHORIZED")

    if hasattr(creds, 'token') and hasattr(creds, 'expired'):
        if creds.expired and creds.refresh_token:
            from google.auth.transport.requests import Request
            creds.refresh(Request())
            _save_token(creds)

        if not creds.token:
            raise PermissionError("NOT_AUTHORIZED")

        _cached_token = {"token": creds.token, "expires": now + 3500}
        return creds.token

    if isinstance(creds, dict):
        token = creds.get("token") or creds.get("access_token", "")
        if token:
            _cached_token = {"token": token, "expires": now + 3500}
            return token

    raise PermissionError("NOT_AUTHORIZED")


def _build_auth_url() -> tuple[str, object]:
    """Build auth URL using InstalledAppFlow. Returns (url, flow)."""
    from google_auth_oauthlib.flow import InstalledAppFlow
    flow = InstalledAppFlow.from_client_secrets_file(str(_creds_path()), SCOPES)
    flow.redirect_uri = "urn:ietf:wg:oauth:2.0:oob"
    auth_url, _ = flow.authorization_url(prompt="consent")
    return auth_url, flow


def _exchange_code(flow, code: str) -> str:
    """Exchange code using the same flow object. Returns success/error message."""
    try:
        flow.fetch_token(code=code.strip())
        _save_token(flow.credentials)
        global _cached_token
        _cached_token = {
            "token": flow.credentials.token,
            "expires": time.time() + 3500,
        }
        return "Google account authorized. /sheets and /docs are now ready."
    except Exception as e:
        log.error(f"Token exchange failed: {e}")
        return f"Authorization failed: {e}"


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------

async def handle(command: str, args: str, secrets: dict, cloud_chat):
    if command == "gauth":
        if args and len(args.strip()) > 10:
            # User pasted a code — find pending flow and exchange
            # flow is stored in _pending_flow by callback_google in berryclaw.py
            # Try all pending flows (there's usually just one)
            for chat_id, flow in list(_pending_flow.items()):
                result = _exchange_code(flow, args)
                _pending_flow.pop(chat_id, None)
                return result
            # No pending flow — create one on the fly
            _, flow = _build_auth_url()
            return _exchange_code(flow, args)

        # Generate auth URL — send as plain text so user can copy to any browser
        url, flow = _build_auth_url()
        _pending_flow[0] = flow
        return (
            "Copy this link and open it in your browser:\n\n"
            f"`{url}`\n\n"
            "Sign in, allow access, then Google shows a code.\n"
            "**Paste that code here as your next message.**"
        )

    # Check auth
    try:
        await _get_access_token()
    except PermissionError:
        buttons = [[InlineKeyboardButton("Authorize Google Account", callback_data="gs:auth")]]
        return {
            "text": "Google not authorized yet. Tap below to connect.",
            "reply_markup": InlineKeyboardMarkup(buttons),
        }
    except Exception as e:
        return f"Google auth error: {e}"

    if command == "sheets":
        return await _handle_sheets(args, cloud_chat)
    elif command == "docs":
        return await _handle_docs(args, cloud_chat)
    return f"Unknown command: {command}"


# ---------------------------------------------------------------------------
# Sheets
# ---------------------------------------------------------------------------

async def _handle_sheets(args: str, cloud_chat):
    if not args:
        authorized = False
        try:
            await _get_access_token()
            authorized = True
        except Exception:
            pass

        if not authorized:
            buttons = [[InlineKeyboardButton("Authorize Google Account", callback_data="gs:auth")]]
            return {
                "text": "**Google Sheets** — Not authorized yet.\nTap below to connect your Google account.",
                "reply_markup": InlineKeyboardMarkup(buttons),
            }

        buttons = [
            [InlineKeyboardButton("Read Sheet", callback_data="gs:read")],
            [InlineKeyboardButton("Write Cell", callback_data="gs:write")],
            [InlineKeyboardButton("Append Row", callback_data="gs:append")],
            [InlineKeyboardButton("List Spreadsheets", callback_data="gs:list")],
            [InlineKeyboardButton("Re-authorize", callback_data="gs:auth")],
        ]
        return {
            "text": "**Google Sheets** — Connected.\nWhat do you want to do?",
            "reply_markup": InlineKeyboardMarkup(buttons),
        }

    parts = args.split(maxsplit=2)
    action = parts[0].lower()

    if action == "list":
        return await _sheets_list()
    if action == "read" and len(parts) >= 2:
        sid = parts[1]
        rng = parts[2] if len(parts) > 2 else "A1:Z100"
        return await _sheets_read(sid, rng)
    if action == "write" and len(parts) >= 3:
        sid = parts[1]
        sub = parts[2].split(maxsplit=1)
        if len(sub) < 2:
            return "Usage: `/sheets write <id> <range> <value>`"
        return await _sheets_write(sid, sub[0], sub[1])
    if action == "append" and len(parts) >= 3:
        sid = parts[1]
        sub = parts[2].split(maxsplit=1)
        if len(sub) < 2:
            return "Usage: `/sheets append <id> <range> val1 | val2 | val3`"
        values = [v.strip() for v in sub[1].split("|")]
        return await _sheets_append(sid, sub[0], values)

    return "Unknown action. Use: list, read, write, append"


def _auth_needed_msg():
    global _cached_token
    _cached_token = {"token": "", "expires": 0}
    # Do NOT delete token.pickle — it may have a valid refresh_token
    buttons = [[InlineKeyboardButton("Re-authorize", callback_data="gs:auth")]]
    return {
        "text": "**Token expired or invalid.** Tap below to re-authorize.",
        "reply_markup": InlineKeyboardMarkup(buttons),
    }


async def _sheets_list():
    token = await _get_access_token()
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(
            "https://www.googleapis.com/drive/v3/files",
            headers={"Authorization": f"Bearer {token}"},
            params={
                "q": "mimeType='application/vnd.google-apps.spreadsheet'",
                "orderBy": "modifiedTime desc",
                "pageSize": 10,
                "fields": "files(id,name,modifiedTime)",
            },
        )
        if r.status_code == 403 and "Drive API" in r.text:
            return (
                "**Drive API not enabled** on this Google Cloud project.\n\n"
                "Enable it here:\n"
                "`https://console.developers.google.com/apis/api/drive.googleapis.com/overview?project=786915304677`\n\n"
                "In the meantime, `/sheets read` and `/sheets write` work fine — just use the sheet ID directly."
            )
        if r.status_code in (401, 403):
            return _auth_needed_msg()
        r.raise_for_status()
        data = r.json()

    files = data.get("files", [])
    if not files:
        return "No spreadsheets found."

    lines = ["**Recent Spreadsheets:**\n"]
    for f in files:
        lines.append(f"- `{f['id'][:12]}...` — {f['name']} ({f.get('modifiedTime', '')[:10]})")
    return "\n".join(lines)


async def _sheets_read(sid: str, rng: str):
    token = await _get_access_token()
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(
            f"{SHEETS_API}/{sid}/values/{rng}",
            headers={"Authorization": f"Bearer {token}"},
        )
        if r.status_code in (401, 403):
            return _auth_needed_msg()
        if r.status_code == 404:
            return "Spreadsheet not found. Check the ID."
        r.raise_for_status()
        data = r.json()

    rows = data.get("values", [])
    if not rows:
        return "No data in that range."
    result = f"**{rng}** ({len(rows)} rows)\n\n"
    for row in rows[:50]:
        result += " | ".join(str(c) for c in row) + "\n"
    if len(rows) > 50:
        result += f"\n... and {len(rows) - 50} more rows"
    return result


async def _sheets_write(sid: str, rng: str, value: str):
    token = await _get_access_token()
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.put(
            f"{SHEETS_API}/{sid}/values/{rng}",
            headers={"Authorization": f"Bearer {token}"},
            params={"valueInputOption": "USER_ENTERED"},
            json={"values": [[value]]},
        )
        if r.status_code in (401, 403):
            return _auth_needed_msg()
        if r.status_code == 404:
            return "Spreadsheet not found."
        r.raise_for_status()
    return f"Written `{value}` to `{rng}`"


async def _sheets_append(sid: str, rng: str, values: list):
    token = await _get_access_token()
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.post(
            f"{SHEETS_API}/{sid}/values/{rng}:append",
            headers={"Authorization": f"Bearer {token}"},
            params={"valueInputOption": "USER_ENTERED", "insertDataOption": "INSERT_ROWS"},
            json={"values": [values]},
        )
        if r.status_code in (401, 403):
            return _auth_needed_msg()
        if r.status_code == 404:
            return "Spreadsheet not found."
        r.raise_for_status()
    return f"Appended row: {' | '.join(values)}"


# ---------------------------------------------------------------------------
# Docs
# ---------------------------------------------------------------------------

async def _handle_docs(args: str, cloud_chat):
    if not args:
        authorized = False
        try:
            await _get_access_token()
            authorized = True
        except Exception:
            pass

        if not authorized:
            buttons = [[InlineKeyboardButton("Authorize Google Account", callback_data="gs:auth")]]
            return {
                "text": "**Google Docs** — Not authorized yet.\nTap below to connect your Google account.",
                "reply_markup": InlineKeyboardMarkup(buttons),
            }

        buttons = [
            [InlineKeyboardButton("Read Document", callback_data="gd:read")],
            [InlineKeyboardButton("Append Text", callback_data="gd:append")],
            [InlineKeyboardButton("Ask About Doc", callback_data="gd:ask")],
            [InlineKeyboardButton("Re-authorize", callback_data="gs:auth")],
        ]
        return {
            "text": "**Google Docs** — What do you want to do?",
            "reply_markup": InlineKeyboardMarkup(buttons),
        }

    parts = args.split(maxsplit=2)
    action = parts[0].lower()

    if action == "read" and len(parts) >= 2:
        return await _docs_read(parts[1])
    if action == "append" and len(parts) >= 3:
        return await _docs_append(parts[1], parts[2])
    if action == "ask" and len(parts) >= 3:
        content = await _docs_read(parts[1])
        if isinstance(content, dict) or (isinstance(content, str) and content.startswith("Error")):
            return content
        answer = await cloud_chat([
            {"role": "system", "content": "Answer based on this document. Be concise."},
            {"role": "user", "content": f"Document:\n{content[:8000]}\n\nQuestion: {parts[2]}"},
        ])
        return answer

    return "Unknown action. Use: read, append, ask"


async def _docs_read(doc_id: str):
    token = await _get_access_token()
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(
            f"{DOCS_API}/{doc_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        if r.status_code in (401, 403):
            return _auth_needed_msg()
        if r.status_code == 404:
            return "Error: Document not found."
        r.raise_for_status()
        doc = r.json()

    title = doc.get("title", "Untitled")
    body = doc.get("body", {}).get("content", [])
    text_parts = []
    for el in body:
        para = el.get("paragraph")
        if para:
            for run in para.get("elements", []):
                tr = run.get("textRun")
                if tr:
                    text_parts.append(tr.get("content", ""))

    content = "".join(text_parts).strip()
    if not content:
        return f"**{title}**\n\n(empty document)"
    if len(content) > 3500:
        content = content[:3500] + "\n\n... (truncated)"
    return f"**{title}**\n\n{content}"


async def _docs_append(doc_id: str, text: str):
    token = await _get_access_token()
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(
            f"{DOCS_API}/{doc_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        if r.status_code in (401, 403):
            return _auth_needed_msg()
        r.raise_for_status()
        doc = r.json()

    end_index = doc.get("body", {}).get("content", [{}])[-1].get("endIndex", 1) - 1

    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.post(
            f"{DOCS_API}/{doc_id}:batchUpdate",
            headers={"Authorization": f"Bearer {token}"},
            json={"requests": [{"insertText": {"location": {"index": max(end_index, 1)}, "text": f"\n{text}"}}]},
        )
        if r.status_code == 404:
            return "Document not found."
        r.raise_for_status()
    return "Appended text to document."
