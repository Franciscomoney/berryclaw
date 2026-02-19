"""Lead Scraper — Button-driven Apify Google Maps → Google Sheets pipeline.

Modes:
  Basic: Single search query → scrape → sheet
  Advanced: Business + City → auto-fetch ZIP codes → batched scraping → sheet

/leads              → Mode selection (Basic / Advanced buttons)
/leads "query"      → Direct basic scrape (backward compat)
/leads enrich ID    → Enrich existing sheet with Firecrawl
/leads status       → Show last scrape stats
"""

import asyncio
import json
import math
import re
import time
import logging
from pathlib import Path

import httpx
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

log = logging.getLogger("berryclaw.leads")

NAME = "leads"
COMMANDS = {
    "leads": "Scrape Google Maps leads and write to Google Sheets",
}
REQUIRED_SECRETS = ["apify_api_key"]
CALLBACK_PREFIX = "leads:"

APIFY_API = "https://api.apify.com/v2"
GOOGLE_MAPS_ACTOR = "franciscoandsam~google-maps-scraper"
ZIPPOPOTAM_API = "http://api.zippopotam.us/us"
BATCH_SIZE = 10  # ZIP codes per actor run

# Fixed column order — NEVER changes, so sheets are always consistent
COLUMNS = [
    "Business Name", "Phone", "Email", "Website", "Address",
    "City", "Rating", "Reviews", "Category", "Google Maps URL",
]

# Default sheet ID — can be overridden per command
DEFAULT_SHEET_ID = "16m1SQsP76wSc7HXsjAfeEDocEbuUDZj8gCxbk6B_J3w"
SHEET_URL = f"https://docs.google.com/spreadsheets/d/{DEFAULT_SHEET_ID}"

# Google workspace creds directory on experiment server
GOOGLE_CREDS_DIR = str(
    Path(__file__).resolve().parent.parent / "google-workspace"
)

# ---------------------------------------------------------------------------
# Session state for button-driven flow
# ---------------------------------------------------------------------------

_sessions: dict[int, dict] = {}
_last_scrape: dict = {}


def has_active_session(chat_id: int) -> bool:
    """Check if a chat has an active leads session awaiting text input."""
    session = _sessions.get(chat_id)
    if not session:
        return False
    # Only intercept text when actually awaiting typed input
    return session.get("step") in ("awaiting_query", "awaiting_business", "awaiting_location")


# ---------------------------------------------------------------------------
# US state abbreviations
# ---------------------------------------------------------------------------

US_STATES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC",
}
VALID_ABBREVS = set(US_STATES.values())


def _normalize_state(raw: str) -> str | None:
    """Convert state input to 2-letter abbreviation. Returns None if invalid."""
    s = raw.strip()
    if len(s) == 2 and s.upper() in VALID_ABBREVS:
        return s.upper()
    full = s.lower()
    if full in US_STATES:
        return US_STATES[full]
    return None


# ---------------------------------------------------------------------------
# Email extraction & validation
# ---------------------------------------------------------------------------

EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    re.IGNORECASE,
)

JUNK_EMAIL_PATTERNS = [
    r".*@example\.com$", r".*@test\.com$", r".*@sentry\.io$",
    r".*noreply@.*", r".*no-reply@.*", r".*@wixpress\.com$",
    r".*@googleusercontent\.com$", r".*@googleapis\.com$",
]
JUNK_RE = [re.compile(p, re.IGNORECASE) for p in JUNK_EMAIL_PATTERNS]
PHONE_RE = re.compile(r"[\+]?[\d\s\-\(\)]{7,20}")


def validate_email(email: str) -> bool:
    """Check if an email looks real (not junk/generated)."""
    if not email or not EMAIL_RE.fullmatch(email):
        return False
    return not any(p.match(email) for p in JUNK_RE)


def extract_emails_from_text(text: str) -> list[str]:
    """Extract all valid emails from a block of text."""
    if not text:
        return []
    return [e for e in EMAIL_RE.findall(text) if validate_email(e)]


def clean_phone(phone: str) -> str:
    """Normalize phone number."""
    if not phone:
        return ""
    cleaned = re.sub(r"[^\d+\-\(\)\s]", "", phone).strip()
    return cleaned if len(re.sub(r"\D", "", cleaned)) >= 7 else ""


# ---------------------------------------------------------------------------
# Apify Google Maps scraping
# ---------------------------------------------------------------------------

async def run_apify_scrape(
    queries: str | list[str],
    api_key: str,
    max_results: int = 100,
    status_callback=None,
    timeout_s: int = 300,
) -> list[dict]:
    """Run the Google Maps scraper. Accepts single query or list of queries."""
    if isinstance(queries, str):
        queries = [queries]

    params = {"token": api_key}
    actor_input = {
        "searchQueries": queries,
        "maxResultsPerQuery": max_results,
        "language": "en",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(
            f"{APIFY_API}/acts/{GOOGLE_MAPS_ACTOR}/runs",
            params=params,
            json=actor_input,
        )
        r.raise_for_status()
        run_id = r.json().get("data", {}).get("id")
        if not run_id:
            raise RuntimeError("Apify: no run ID returned")
        if status_callback:
            await status_callback(f"Apify run started (ID: {run_id[:8]}...). Scraping...")

    # Poll for completion
    max_polls = max(timeout_s // 5, 12)
    async with httpx.AsyncClient(timeout=30.0) as client:
        for i in range(max_polls):
            await asyncio.sleep(5)
            r = await client.get(
                f"{APIFY_API}/actor-runs/{run_id}",
                params=params,
            )
            r.raise_for_status()
            status = r.json().get("data", {}).get("status")

            if status == "SUCCEEDED":
                dataset_id = r.json().get("data", {}).get("defaultDatasetId")
                if not dataset_id:
                    raise RuntimeError("Apify: succeeded but no dataset")
                r2 = await client.get(
                    f"{APIFY_API}/datasets/{dataset_id}/items",
                    params={**params, "format": "json"},
                )
                r2.raise_for_status()
                items = r2.json()
                if status_callback:
                    await status_callback(f"Done. {len(items)} results. Processing...")
                return items

            elif status in ("FAILED", "ABORTED", "TIMED-OUT"):
                raise RuntimeError(f"Apify run {status.lower()}")

            if status_callback and i > 0 and i % 6 == 0:
                await status_callback(f"Still scraping... ({(i + 1) * 5}s elapsed)")

    raise RuntimeError(f"Apify run timed out after {timeout_s}s")


# ---------------------------------------------------------------------------
# Parse Apify results into clean rows
# ---------------------------------------------------------------------------

def parse_results(items: list[dict]) -> list[list[str]]:
    """Convert raw Apify items into clean sheet rows. DETERMINISTIC — no LLM."""
    rows = []
    seen_names = set()

    for item in items:
        name = (item.get("title") or item.get("name") or "").strip()
        if not name or name.lower() in seen_names:
            continue
        seen_names.add(name.lower())

        # Phone
        phone = clean_phone(item.get("phone") or item.get("phoneUnformatted") or "")

        # Email — try direct field first
        email = ""
        direct_email = item.get("email") or item.get("emails") or ""
        if isinstance(direct_email, list):
            valid = [e for e in direct_email if validate_email(e)]
            email = valid[0] if valid else ""
        elif isinstance(direct_email, str) and validate_email(direct_email):
            email = direct_email

        # If no direct email, try extracting from text fields
        if not email:
            for src in [
                item.get("website") or "",
                item.get("description") or "",
                json.dumps(item.get("additionalInfo") or {}),
            ]:
                found = extract_emails_from_text(src)
                if found:
                    email = found[0]
                    break

        # Website vs Google Maps URL
        website = (item.get("website") or item.get("url") or "").strip()
        maps_url = (item.get("url") or item.get("googleMapsUrl") or "").strip()
        if website == maps_url:
            website = ""

        # Address & City
        address = (item.get("address") or item.get("street") or "").strip()
        city = (item.get("city") or item.get("neighborhood") or "").strip()
        if not city and address:
            parts = address.rsplit(",", 2)
            if len(parts) >= 2:
                city = parts[-2].strip() if len(parts) >= 3 else parts[-1].strip()

        rating = str(item.get("totalScore") or item.get("rating") or "")
        reviews = str(item.get("reviewsCount") or item.get("reviews") or "")
        category = (item.get("categoryName") or item.get("category") or "").strip()

        rows.append([name, phone, email, website, address, city, rating, reviews, category, maps_url])

    return rows


# ---------------------------------------------------------------------------
# Google Sheets writing
# ---------------------------------------------------------------------------

SHEETS_API = "https://sheets.googleapis.com/v4/spreadsheets"


async def get_sheets_token(creds_dir: str) -> str:
    """Load OAuth token from token.pickle."""
    import pickle
    token_path = Path(creds_dir) / "token.pickle"
    if not token_path.exists():
        raise FileNotFoundError(
            f"No token.pickle at {token_path}. "
            "Run google_auth.py on the server first to create it."
        )
    with open(token_path, "rb") as f:
        creds = pickle.load(f)
    if creds.expired and creds.refresh_token:
        from google.auth.transport.requests import Request
        creds.refresh(Request())
        with open(token_path, "wb") as f:
            pickle.dump(creds, f)
    return creds.token


async def write_to_sheets(
    rows: list[list[str]],
    spreadsheet_id: str,
    creds_dir: str,
    sheet_name: str = "Leads",
) -> dict:
    """Write lead rows to Google Sheets (clears existing data first)."""
    token = await get_sheets_token(creds_dir)
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Check if sheet/tab exists, create if not
        r = await client.get(f"{SHEETS_API}/{spreadsheet_id}", headers=headers)
        r.raise_for_status()
        titles = [s["properties"]["title"] for s in r.json().get("sheets", [])]

        if sheet_name not in titles:
            await client.post(
                f"{SHEETS_API}/{spreadsheet_id}:batchUpdate",
                headers=headers,
                json={"requests": [{"addSheet": {"properties": {"title": sheet_name}}}]},
            )

        # Clear existing data
        await client.post(
            f"{SHEETS_API}/{spreadsheet_id}/values/{sheet_name}:clear",
            headers=headers,
            json={},
        )

        # Write header + rows
        r = await client.put(
            f"{SHEETS_API}/{spreadsheet_id}/values/{sheet_name}!A1",
            headers=headers,
            params={"valueInputOption": "USER_ENTERED"},
            json={"values": [COLUMNS] + rows},
        )
        r.raise_for_status()
        update_data = r.json()

    return {
        "rows_written": len(rows),
        "total_cells": update_data.get("updatedCells", 0),
        "sheet_name": sheet_name,
    }


async def append_to_sheets(
    rows: list[list[str]],
    spreadsheet_id: str,
    creds_dir: str,
    sheet_name: str = "Leads",
) -> dict:
    """Append lead rows to existing sheet (doesn't clear)."""
    token = await get_sheets_token(creds_dir)
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(
            f"{SHEETS_API}/{spreadsheet_id}/values/{sheet_name}!A:J:append",
            headers=headers,
            params={"valueInputOption": "USER_ENTERED", "insertDataOption": "INSERT_ROWS"},
            json={"values": rows},
        )
        r.raise_for_status()

    return {"rows_appended": len(rows), "sheet_name": sheet_name}


# ---------------------------------------------------------------------------
# Firecrawl email enrichment
# ---------------------------------------------------------------------------

async def enrich_with_firecrawl(
    rows: list[list[str]],
    firecrawl_key: str,
    status_callback=None,
) -> tuple[list[list[str]], int]:
    """Scrape websites from rows to find missing emails."""
    if not firecrawl_key:
        return rows, 0

    found = 0
    headers = {
        "Authorization": f"Bearer {firecrawl_key}",
        "Content-Type": "application/json",
    }

    for i, row in enumerate(rows):
        website = row[3]  # Website column
        email = row[2]    # Email column
        if email or not website:
            continue

        if status_callback and i % 10 == 0:
            await status_callback(f"Enriching... ({i}/{len(rows)}, {found} emails found)")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.post(
                    "https://api.firecrawl.dev/v1/scrape",
                    headers=headers,
                    json={"url": website, "formats": ["markdown"]},
                )
                if r.status_code == 200:
                    md = r.json().get("data", {}).get("markdown", "")
                    emails = extract_emails_from_text(md)
                    if emails:
                        row[2] = emails[0]
                        found += 1
            await asyncio.sleep(1)
        except Exception as e:
            log.warning("Firecrawl failed for %s: %s", website, e)

    return rows, found


# ---------------------------------------------------------------------------
# ZIP code fetching
# ---------------------------------------------------------------------------

async def _fetch_zip_codes(city: str, state: str) -> list[str]:
    """Fetch ZIP codes for a city from Zippopotam API."""
    state_lower = state.lower()
    # URL-encode spaces in city name
    city_url = city.replace(" ", "%20").lower()
    url = f"{ZIPPOPOTAM_API}/{state_lower}/{city_url}"

    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(url)
        if r.status_code == 404:
            return []
        r.raise_for_status()
        data = r.json()

    places = data.get("places", [])

    # Filter by exact city name match (API sometimes returns nearby places)
    city_lower = city.lower()
    zips = [
        p["post code"]
        for p in places
        if p.get("place name", "").lower() == city_lower
    ]

    # If exact match returns nothing, use all results
    if not zips:
        zips = [p["post code"] for p in places]

    return sorted(set(zips))


# ---------------------------------------------------------------------------
# Button flow: handle (entry) + callback (buttons) + handle_text (text input)
# ---------------------------------------------------------------------------

async def handle(command: str, args: str, secrets: dict, cloud_chat) -> str | dict:
    """Handle /leads commands."""

    if not args:
        # Show mode selection buttons
        return {
            "text": (
                "**Lead Scraper**\n\n"
                "**Basic** — Type a search query (e.g. \"plumbers in Miami\"). "
                "Gets ~100 results from Google Maps.\n\n"
                "**Advanced** — Pick a business type + city. "
                "Auto-splits into ZIP codes for full coverage (500+ leads).\n\n"
                f"Results go to [Lista001]({SHEET_URL}) — "
                "each run creates a new tab."
            ),
            "reply_markup": InlineKeyboardMarkup([[
                InlineKeyboardButton("Basic", callback_data="leads:basic"),
                InlineKeyboardButton("Advanced", callback_data="leads:advanced"),
            ]]),
        }

    args_lower = args.strip().lower()

    if args_lower.startswith("enrich"):
        return await _handle_enrich(args, secrets)

    if args_lower == "status":
        return _get_status()

    # Direct query — backward compatible basic scrape
    return await _handle_scrape(args, secrets)


async def callback(query, secrets: dict):
    """Handle button presses (leads:xxx). Called from berryclaw callback handler."""
    data = query.data
    chat_id = query.message.chat_id
    action = data.split(":", 1)[1] if ":" in data else ""

    if action == "basic":
        _sessions[chat_id] = {"step": "awaiting_query", "mode": "basic"}
        await query.edit_message_text(
            "**Basic Mode**\n\n"
            "Type your search query exactly as you'd search Google Maps.\n\n"
            "Example: `barbershops in Las Vegas`\n\n"
            "I'll scrape ~100 results and write them to a new tab "
            "in Lista001 (named after your query).",
            parse_mode="Markdown",
        )

    elif action == "advanced":
        _sessions[chat_id] = {"step": "awaiting_business", "mode": "advanced"}
        await query.edit_message_text(
            "**Advanced Mode — Step 1/3**\n\n"
            "What type of business are you looking for?\n\n"
            "Example: `barbershops`, `plumbers`, `dentists`",
            parse_mode="Markdown",
        )

    elif action == "start":
        session = _sessions.pop(chat_id, None)
        if not session or not session.get("zips"):
            await query.edit_message_text("Session expired. Send /leads to start over.")
            return

        business = session["business"]
        city = session.get("city", "")
        state = session.get("state", "")
        zips = session["zips"]

        async def progress(msg):
            try:
                await query.edit_message_text(msg, parse_mode="Markdown")
            except Exception:
                pass

        try:
            result = await _run_advanced(business, city, state, zips, secrets, progress)
            try:
                await query.edit_message_text(result, parse_mode="Markdown")
            except Exception:
                await query.edit_message_text(result)
        except Exception as e:
            log.error("Advanced scrape failed: %s", e)
            await query.edit_message_text(f"Pipeline error: {e}")

    elif action == "cancel":
        _sessions.pop(chat_id, None)
        await query.edit_message_text("Cancelled.")

    else:
        await query.edit_message_text(f"Unknown action: {action}")


async def handle_text(chat_id: int, text: str, secrets: dict) -> str | dict | None:
    """Handle plain text input during active session. Returns result to display."""
    session = _sessions.get(chat_id)
    if not session:
        return None

    step = session["step"]

    if step == "awaiting_query":
        # Basic mode — run scrape with this query
        _sessions.pop(chat_id)
        return await _handle_scrape(text.strip(), secrets)

    elif step == "awaiting_business":
        session["business"] = text.strip()
        session["step"] = "awaiting_location"
        return (
            "**Step 2/3 — Location**\n\n"
            f"Business: **{text.strip()}**\n\n"
            "Now enter the city and state:\n\n"
            "Example: `Las Vegas, NV` or `Miami, FL`"
        )

    elif step == "awaiting_location":
        result = _parse_location(text.strip())
        if result is None:
            return (
                "Couldn't parse that. Use format: `City, ST`\n\n"
                "Example: `Las Vegas, NV` or `Miami, FL`"
            )

        city, state = result

        try:
            zips = await _fetch_zip_codes(city, state)
        except Exception as e:
            _sessions.pop(chat_id, None)
            return f"Failed to fetch ZIP codes: {e}"

        if not zips:
            _sessions.pop(chat_id, None)
            return (
                f"No ZIP codes found for **{city}, {state}**.\n\n"
                "Check the city name and try again with /leads"
            )

        session["city"] = city
        session["state"] = state
        session["zips"] = zips
        session["step"] = "confirm"

        batches = math.ceil(len(zips) / BATCH_SIZE)
        business = session["business"]
        tab_name = f"{business[:20]} {city[:10]} {time.strftime('%m-%d')}"
        est_minutes = batches * 3  # ~3 min per batch

        return {
            "text": (
                f"**Step 3/3 — Confirm**\n\n"
                f"**Business:** {business}\n"
                f"**Location:** {city}, {state}\n"
                f"**ZIP codes:** {len(zips)}\n"
                f"**Batches:** {batches} ({BATCH_SIZE} ZIPs each)\n"
                f"**Est. time:** ~{est_minutes} min\n\n"
                f"Results will be written to:\n"
                f"[Lista001]({SHEET_URL}) → tab `{tab_name}`\n\n"
                f"ZIPs: `{', '.join(zips[:15])}{'...' if len(zips) > 15 else ''}`"
            ),
            "reply_markup": InlineKeyboardMarkup([[
                InlineKeyboardButton("Start Scraping", callback_data="leads:start"),
                InlineKeyboardButton("Cancel", callback_data="leads:cancel"),
            ]]),
        }

    return None


def _parse_location(text: str) -> tuple[str, str] | None:
    """Parse 'City, ST' or 'City, State Name' into (city, state_abbrev)."""
    if "," not in text:
        return None
    parts = text.rsplit(",", 1)
    city = parts[0].strip()
    state_raw = parts[1].strip()
    if not city or not state_raw:
        return None
    state = _normalize_state(state_raw)
    if not state:
        return None
    return city, state


# ---------------------------------------------------------------------------
# Core scrape logic
# ---------------------------------------------------------------------------

async def _handle_scrape(args: str, secrets: dict) -> str:
    """Full basic pipeline: parse args → scrape → validate → write to sheets."""
    api_key = secrets.get("apify_api_key", "")
    if not api_key:
        return "Apify API key not configured."

    # Parse: "query" [sheet_id] [max_results]
    parts = args.strip().split()
    query = ""
    sheet_id = DEFAULT_SHEET_ID
    max_results = 100

    if args.strip().startswith('"'):
        match = re.match(r'"([^"]+)"(.*)', args.strip())
        if match:
            query = match.group(1)
            rest = match.group(2).strip().split()
            if rest:
                sheet_id = rest[0]
            if len(rest) > 1:
                try:
                    max_results = int(rest[1])
                except ValueError:
                    pass
    else:
        if len(parts) >= 2 and len(parts[-1]) > 20:
            sheet_id = parts[-1]
            query = " ".join(parts[:-1])
        elif len(parts) >= 3 and parts[-1].isdigit():
            max_results = int(parts[-1])
            if len(parts[-2]) > 20:
                sheet_id = parts[-2]
                query = " ".join(parts[:-2])
            else:
                query = " ".join(parts[:-1])
        else:
            query = " ".join(parts)

    if not query:
        return "Please provide a search query."

    start_time = time.time()

    try:
        items = await run_apify_scrape(query, api_key, max_results)
        if not items:
            return f"No results for: {query}"

        rows = parse_results(items)
        if not rows:
            return f"Scraped {len(items)} items but none had usable data."

        # Enrich with Firecrawl if available
        firecrawl_key = secrets.get("firecrawl_api_key", "")
        enriched_count = 0
        if firecrawl_key:
            no_email = [r for r in rows if not r[2] and r[3]]
            if no_email:
                _, enriched_count = await enrich_with_firecrawl(no_email[:20], firecrawl_key)

        # Write to Google Sheets
        sheet_name = query[:30].replace('"', '').strip()
        try:
            stats = await write_to_sheets(rows, sheet_id, GOOGLE_CREDS_DIR, sheet_name)
            sheet_status = f"Written: {stats['rows_written']} rows → `{sheet_name}`"
        except Exception as e:
            log.error("Sheets write failed: %s", e)
            sheet_status = f"Sheet write FAILED: {e}"

        duration = round(time.time() - start_time, 1)
        with_email = sum(1 for r in rows if r[2])
        with_phone = sum(1 for r in rows if r[1])
        with_website = sum(1 for r in rows if r[3])

        _last_scrape.update({
            "query": query,
            "total_scraped": len(items),
            "parsed": len(rows),
            "with_email": with_email,
            "with_phone": with_phone,
            "rows_written": len(rows),
            "duration": duration,
        })

        report = (
            f"**Lead Scrape Complete**\n\n"
            f"**Query:** {query}\n"
            f"**Results:** {len(rows)} leads (from {len(items)} raw)\n\n"
            f"**Data quality:**\n"
            f"  Phone: {with_phone}/{len(rows)} ({_pct(with_phone, len(rows))})\n"
            f"  Email: {with_email}/{len(rows)} ({_pct(with_email, len(rows))})\n"
            f"  Website: {with_website}/{len(rows)} ({_pct(with_website, len(rows))})\n"
        )

        if enriched_count:
            report += f"  Enriched: +{enriched_count} emails\n"

        report += (
            f"\n**Sheet:** {sheet_status}\n"
            f"[Open spreadsheet]({SHEET_URL})\n"
            f"**Time:** {duration}s"
        )
        return report

    except Exception as e:
        log.error("Lead scrape failed: %s", e)
        return f"Pipeline error: {e}"


async def _run_advanced(
    business: str,
    city: str,
    state: str,
    zips: list[str],
    secrets: dict,
    progress_callback=None,
) -> str:
    """Run batched ZIP code scraping for full city coverage."""
    api_key = secrets.get("apify_api_key", "")
    if not api_key:
        return "Apify API key not configured."

    batches = [zips[i:i + BATCH_SIZE] for i in range(0, len(zips), BATCH_SIZE)]
    all_items = []
    start_time = time.time()

    for batch_idx, batch_zips in enumerate(batches):
        queries = [f"{business} {z}" for z in batch_zips]

        if progress_callback:
            preview = ", ".join(batch_zips[:5])
            if len(batch_zips) > 5:
                preview += f"... +{len(batch_zips) - 5} more"
            await progress_callback(
                f"**Batch {batch_idx + 1}/{len(batches)}**\n"
                f"Scraping {len(batch_zips)} ZIP codes: {preview}\n"
                f"Total raw so far: {len(all_items)}"
            )

        try:
            items = await run_apify_scrape(
                queries, api_key, max_results=80, timeout_s=600,
            )
            all_items.extend(items)
        except Exception as e:
            log.error("Batch %d failed: %s", batch_idx + 1, e)
            if progress_callback:
                await progress_callback(
                    f"**Batch {batch_idx + 1} error:** {e}\nContinuing..."
                )
            await asyncio.sleep(3)

    if not all_items:
        return "No results found across any batch."

    if progress_callback:
        await progress_callback(f"Processing {len(all_items)} raw results...")

    # Parse & dedup all at once
    rows = parse_results(all_items)

    if not rows:
        return f"Scraped {len(all_items)} items but none had usable data."

    # Write to sheet
    sheet_name = f"{business[:20]} {city[:10]} {time.strftime('%m-%d')}"
    try:
        stats = await write_to_sheets(rows, DEFAULT_SHEET_ID, GOOGLE_CREDS_DIR, sheet_name)
        sheet_msg = f"Written: {stats['rows_written']} rows → `{sheet_name}`"
    except Exception as e:
        log.error("Sheets write failed: %s", e)
        sheet_msg = f"Sheet write FAILED: {e}"

    duration = round(time.time() - start_time, 1)
    with_email = sum(1 for r in rows if r[2])
    with_phone = sum(1 for r in rows if r[1])

    _last_scrape.update({
        "query": f"{business} in {city}, {state} (advanced)",
        "total_scraped": len(all_items),
        "parsed": len(rows),
        "with_email": with_email,
        "with_phone": with_phone,
        "rows_written": len(rows),
        "duration": duration,
    })

    return (
        f"**Advanced Scrape Complete**\n\n"
        f"**Business:** {business}\n"
        f"**Location:** {city}, {state}\n"
        f"**ZIP codes:** {len(zips)} across {len(batches)} batches\n"
        f"**Unique leads:** {len(rows)} (from {len(all_items)} raw)\n\n"
        f"**Data quality:**\n"
        f"  Phone: {with_phone}/{len(rows)} ({_pct(with_phone, len(rows))})\n"
        f"  Email: {with_email}/{len(rows)} ({_pct(with_email, len(rows))})\n\n"
        f"**Sheet:** {sheet_msg}\n"
        f"[Open spreadsheet]({SHEET_URL})\n"
        f"**Time:** {duration}s"
    )


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def _get_status() -> str:
    if not _last_scrape:
        return "No scrapes yet this session."
    s = _last_scrape
    return (
        f"**Last scrape:**\n"
        f"  Query: {s.get('query', '?')}\n"
        f"  Results: {s.get('total_scraped', 0)}\n"
        f"  With email: {s.get('with_email', 0)}\n"
        f"  With phone: {s.get('with_phone', 0)}\n"
        f"  Sheet rows: {s.get('rows_written', 0)}\n"
        f"  Time: {s.get('duration', '?')}s"
    )


async def _handle_enrich(args: str, secrets: dict) -> str:
    """Enrich existing sheet data with emails from Firecrawl."""
    firecrawl_key = secrets.get("firecrawl_api_key", "")
    if not firecrawl_key:
        return "Firecrawl API key not configured. Add it with `/api set firecrawl_api_key KEY`"

    parts = args.strip().split()
    sheet_id = parts[1] if len(parts) > 1 else DEFAULT_SHEET_ID
    sheet_name = parts[2] if len(parts) > 2 else "Leads"

    try:
        token = await get_sheets_token(GOOGLE_CREDS_DIR)
        headers = {"Authorization": f"Bearer {token}"}

        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(
                f"{SHEETS_API}/{sheet_id}/values/{sheet_name}!A:J",
                headers=headers,
            )
            r.raise_for_status()
            values = r.json().get("values", [])

        if len(values) <= 1:
            return "Sheet is empty or has only headers."

        rows = [r + [""] * (10 - len(r)) for r in values[1:]]
        rows, found = await enrich_with_firecrawl(rows, firecrawl_key)

        if found:
            await write_to_sheets(rows, sheet_id, GOOGLE_CREDS_DIR, sheet_name)
            return f"Enrichment complete. Found {found} new emails from {len(rows)} websites."
        return "No new emails found via website scraping."

    except Exception as e:
        return f"Enrichment error: {e}"


def _pct(n: int, total: int) -> str:
    if total == 0:
        return "0%"
    return f"{round(n / total * 100)}%"
