"""Google Workspace integration — read/write Google Sheets and Docs."""

import json

import httpx

NAME = "google"
COMMANDS = {
    "sheets": "Read or write Google Sheets",
    "docs": "Read or edit Google Docs",
}
REQUIRED_SECRETS = ["google_credentials_file"]

# Google API endpoints
SHEETS_API = "https://sheets.googleapis.com/v4/spreadsheets"
DOCS_API = "https://docs.googleapis.com/v1/documents"
TOKEN_URL = "https://oauth2.googleapis.com/token"

_cached_token: dict = {"token": "", "expires": 0}


async def handle(command: str, args: str, secrets: dict, cloud_chat) -> str:
    creds_file = secrets.get("google_credentials_file", "")
    if not creds_file:
        return "Google credentials not configured in secrets.json."

    if command == "sheets":
        return await _handle_sheets(args, creds_file, cloud_chat)
    elif command == "docs":
        return await _handle_docs(args, creds_file, cloud_chat)

    return f"Unknown command: {command}"


async def _handle_sheets(args: str, creds_file: str, cloud_chat) -> str:
    if not args:
        return (
            "Usage:\n"
            "  `/sheets read <spreadsheet_id> [range]`\n"
            "  `/sheets write <spreadsheet_id> <range> <value>`\n"
            "  `/sheets append <spreadsheet_id> <range> <values...>`\n\n"
            "Example:\n"
            "  `/sheets read 1BxiMVs A1:D10`\n"
            "  `/sheets write 1BxiMVs A1 Hello`\n"
            "  `/sheets append 1BxiMVs A:D val1 | val2 | val3`"
        )

    parts = args.split(maxsplit=2)
    action = parts[0].lower()

    if action == "read" and len(parts) >= 2:
        spreadsheet_id = parts[1]
        cell_range = parts[2] if len(parts) > 2 else "A1:Z100"
        return await _sheets_read(spreadsheet_id, cell_range, creds_file)

    elif action == "write" and len(parts) >= 3:
        sub_parts = parts[2].split(maxsplit=1)
        if len(sub_parts) < 2:
            return "Usage: `/sheets write <id> <range> <value>`"
        spreadsheet_id = parts[1]
        cell_range = sub_parts[0]
        value = sub_parts[1]
        return await _sheets_write(spreadsheet_id, cell_range, value, creds_file)

    elif action == "append" and len(parts) >= 3:
        sub_parts = parts[2].split(maxsplit=1)
        if len(sub_parts) < 2:
            return "Usage: `/sheets append <id> <range> val1 | val2 | val3`"
        spreadsheet_id = parts[1]
        cell_range = sub_parts[0]
        values = [v.strip() for v in sub_parts[1].split("|")]
        return await _sheets_append(spreadsheet_id, cell_range, values, creds_file)

    return "Unknown sheets action. Use: read, write, append"


async def _handle_docs(args: str, creds_file: str, cloud_chat) -> str:
    if not args:
        return (
            "Usage:\n"
            "  `/docs read <doc_id>`\n"
            "  `/docs append <doc_id> <text>`\n"
            "  `/docs ask <doc_id> <question>`\n\n"
            "Example:\n"
            "  `/docs read 1abc123`\n"
            "  `/docs append 1abc123 New paragraph here`\n"
            "  `/docs ask 1abc123 What is this document about?`"
        )

    parts = args.split(maxsplit=2)
    action = parts[0].lower()

    if action == "read" and len(parts) >= 2:
        doc_id = parts[1]
        return await _docs_read(doc_id, creds_file)

    elif action == "append" and len(parts) >= 3:
        doc_id = parts[1]
        text = parts[2]
        return await _docs_append(doc_id, text, creds_file)

    elif action == "ask" and len(parts) >= 3:
        doc_id = parts[1]
        question = parts[2]
        content = await _docs_read(doc_id, creds_file)
        if content.startswith("Error"):
            return content
        answer = await cloud_chat([
            {"role": "system", "content": "Answer based on this document. Be concise."},
            {"role": "user", "content": f"Document:\n{content[:8000]}\n\nQuestion: {question}"},
        ])
        return answer

    return "Unknown docs action. Use: read, append, ask"


# ---------------------------------------------------------------------------
# Google Auth — service account JWT → access token
# ---------------------------------------------------------------------------

async def _get_token(creds_file: str) -> str:
    """Get an access token from a service account JSON file."""
    import time

    global _cached_token
    now = time.time()
    if _cached_token["token"] and _cached_token["expires"] > now + 60:
        return _cached_token["token"]

    from pathlib import Path
    creds_path = Path(creds_file)
    if not creds_path.is_absolute():
        creds_path = Path(__file__).resolve().parent.parent / creds_file

    if not creds_path.exists():
        raise FileNotFoundError(f"Credentials file not found: {creds_path}")

    with open(creds_path) as f:
        creds = json.load(f)

    # Build JWT
    import base64
    import hashlib

    header = base64.urlsafe_b64encode(json.dumps(
        {"alg": "RS256", "typ": "JWT"}
    ).encode()).rstrip(b"=").decode()

    iat = int(now)
    exp = iat + 3600
    claim_set = {
        "iss": creds["client_email"],
        "scope": "https://www.googleapis.com/auth/spreadsheets https://www.googleapis.com/auth/documents",
        "aud": TOKEN_URL,
        "iat": iat,
        "exp": exp,
    }
    payload = base64.urlsafe_b64encode(
        json.dumps(claim_set).encode()
    ).rstrip(b"=").decode()

    signing_input = f"{header}.{payload}"

    # Sign with RSA private key
    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding

        private_key = serialization.load_pem_private_key(
            creds["private_key"].encode(), password=None
        )
        signature = private_key.sign(
            signing_input.encode(),
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
        sig_b64 = base64.urlsafe_b64encode(signature).rstrip(b"=").decode()
    except ImportError:
        raise ImportError(
            "Google integration requires 'cryptography' package. "
            "Install with: pip3 install cryptography"
        )

    jwt_token = f"{signing_input}.{sig_b64}"

    # Exchange JWT for access token
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.post(TOKEN_URL, data={
            "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
            "assertion": jwt_token,
        })
        r.raise_for_status()
        token_data = r.json()

    _cached_token = {
        "token": token_data["access_token"],
        "expires": now + token_data.get("expires_in", 3600),
    }
    return _cached_token["token"]


# ---------------------------------------------------------------------------
# Sheets operations
# ---------------------------------------------------------------------------

async def _sheets_read(spreadsheet_id: str, cell_range: str, creds_file: str) -> str:
    token = await _get_token(creds_file)
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(
            f"{SHEETS_API}/{spreadsheet_id}/values/{cell_range}",
            headers={"Authorization": f"Bearer {token}"},
        )
        if r.status_code == 404:
            return "Spreadsheet not found. Make sure it's shared with the service account email."
        r.raise_for_status()
        data = r.json()

    rows = data.get("values", [])
    if not rows:
        return "No data found in that range."

    # Format as table
    result = f"**Range:** {cell_range} ({len(rows)} rows)\n\n"
    for i, row in enumerate(rows[:50]):
        result += " | ".join(str(cell) for cell in row) + "\n"

    return result


async def _sheets_write(spreadsheet_id: str, cell_range: str, value: str, creds_file: str) -> str:
    token = await _get_token(creds_file)
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.put(
            f"{SHEETS_API}/{spreadsheet_id}/values/{cell_range}",
            headers={"Authorization": f"Bearer {token}"},
            params={"valueInputOption": "USER_ENTERED"},
            json={"values": [[value]]},
        )
        if r.status_code == 404:
            return "Spreadsheet not found. Make sure it's shared with the service account email."
        r.raise_for_status()

    return f"Written `{value}` to `{cell_range}`"


async def _sheets_append(spreadsheet_id: str, cell_range: str, values: list, creds_file: str) -> str:
    token = await _get_token(creds_file)
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.post(
            f"{SHEETS_API}/{spreadsheet_id}/values/{cell_range}:append",
            headers={"Authorization": f"Bearer {token}"},
            params={"valueInputOption": "USER_ENTERED", "insertDataOption": "INSERT_ROWS"},
            json={"values": [values]},
        )
        if r.status_code == 404:
            return "Spreadsheet not found. Make sure it's shared with the service account email."
        r.raise_for_status()

    return f"Appended row: {' | '.join(values)}"


# ---------------------------------------------------------------------------
# Docs operations
# ---------------------------------------------------------------------------

async def _docs_read(doc_id: str, creds_file: str) -> str:
    token = await _get_token(creds_file)
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(
            f"{DOCS_API}/{doc_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        if r.status_code == 404:
            return "Document not found. Make sure it's shared with the service account email."
        r.raise_for_status()
        doc = r.json()

    # Extract text from document body
    title = doc.get("title", "Untitled")
    body = doc.get("body", {}).get("content", [])
    text_parts = []

    for element in body:
        paragraph = element.get("paragraph")
        if paragraph:
            for run in paragraph.get("elements", []):
                text_run = run.get("textRun")
                if text_run:
                    text_parts.append(text_run.get("content", ""))

    content = "".join(text_parts).strip()
    if not content:
        return f"**{title}**\n\n(empty document)"

    if len(content) > 3500:
        content = content[:3500] + "\n\n... (truncated)"

    return f"**{title}**\n\n{content}"


async def _docs_append(doc_id: str, text: str, creds_file: str) -> str:
    token = await _get_token(creds_file)

    # First get the document to find the end index
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(
            f"{DOCS_API}/{doc_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        r.raise_for_status()
        doc = r.json()

    body = doc.get("body", {})
    end_index = body.get("content", [{}])[-1].get("endIndex", 1) - 1

    # Append text
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.post(
            f"{DOCS_API}/{doc_id}:batchUpdate",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "requests": [{
                    "insertText": {
                        "location": {"index": max(end_index, 1)},
                        "text": f"\n{text}",
                    }
                }]
            },
        )
        if r.status_code == 404:
            return "Document not found. Make sure it's shared with the service account email."
        r.raise_for_status()

    return f"Appended text to document."
