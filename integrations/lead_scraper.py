"""Lead Scraper — deterministic Apify Google Maps → validate → Google Sheets pipeline.

This is the KEY integration. It does NOT rely on the LLM for data handling.
The pipeline is:
  1. User says: /leads "plumbers in Miami" [sheet_id]
  2. Apify Google Maps scraper runs with that query
  3. Results are parsed deterministically (fixed field extraction)
  4. Emails/phones are validated with regex
  5. Data is written to Google Sheets with consistent columns
  6. Bot reports back with summary

Optional enrichment:
  /leads enrich <sheet_id> — scrape websites from column to find emails via Firecrawl
"""

import asyncio
import json
import re
import time
import logging
from pathlib import Path

import httpx

log = logging.getLogger("berryclaw.leads")

NAME = "leads"
COMMANDS = {
    "leads": "Scrape Google Maps leads and write to Google Sheets",
}
REQUIRED_SECRETS = ["apify_api_key"]

APIFY_API = "https://api.apify.com/v2"
GOOGLE_MAPS_ACTOR = "franciscoandsam~google-maps-scraper"

# Fixed column order — NEVER changes, so sheets are always consistent
COLUMNS = [
    "Business Name",
    "Phone",
    "Email",
    "Website",
    "Address",
    "City",
    "Rating",
    "Reviews",
    "Category",
    "Google Maps URL",
]

# ---------------------------------------------------------------------------
# Email extraction & validation
# ---------------------------------------------------------------------------

EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    re.IGNORECASE,
)

# Junk emails to filter out (generic, not real contacts)
JUNK_EMAIL_PATTERNS = [
    r".*@example\.com$",
    r".*@test\.com$",
    r".*@sentry\.io$",
    r".*noreply@.*",
    r".*no-reply@.*",
    r".*@wixpress\.com$",
    r".*@googleusercontent\.com$",
    r".*@googleapis\.com$",
]

JUNK_RE = [re.compile(p, re.IGNORECASE) for p in JUNK_EMAIL_PATTERNS]

PHONE_RE = re.compile(r"[\+]?[\d\s\-\(\)]{7,20}")


def validate_email(email: str) -> bool:
    """Check if an email looks real (not junk/generated)."""
    if not email or not EMAIL_RE.fullmatch(email):
        return False
    for pattern in JUNK_RE:
        if pattern.match(email):
            return False
    return True


def extract_emails_from_text(text: str) -> list[str]:
    """Extract all valid emails from a block of text."""
    if not text:
        return []
    found = EMAIL_RE.findall(text)
    return [e for e in found if validate_email(e)]


def clean_phone(phone: str) -> str:
    """Normalize phone number."""
    if not phone:
        return ""
    # Remove extra whitespace, keep digits and +
    cleaned = re.sub(r"[^\d+\-\(\)\s]", "", phone).strip()
    return cleaned if len(re.sub(r"\D", "", cleaned)) >= 7 else ""


# ---------------------------------------------------------------------------
# Apify Google Maps scraping
# ---------------------------------------------------------------------------

async def run_apify_scrape(query: str, api_key: str, max_results: int = 100,
                           status_callback=None) -> list[dict]:
    """Run the Google Maps scraper and return structured results."""
    headers = {"Authorization": f"Bearer {api_key}"}

    # Build input for the actor
    actor_input = {
        "searchStringsArray": [query],
        "maxCrawledPlacesPerSearch": max_results,
        "language": "en",
        "includeWebResults": False,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Start the actor run
        r = await client.post(
            f"{APIFY_API}/acts/{GOOGLE_MAPS_ACTOR}/runs",
            headers=headers,
            json=actor_input,
        )
        r.raise_for_status()
        run_data = r.json().get("data", {})
        run_id = run_data.get("id")

        if not run_id:
            raise RuntimeError("Apify: actor started but no run ID returned")

        if status_callback:
            await status_callback(f"Apify run started (ID: {run_id[:8]}...). Scraping...")

    # Poll for completion — Google Maps can take a while for large queries
    async with httpx.AsyncClient(timeout=30.0) as client:
        max_polls = 60  # up to 5 minutes
        for i in range(max_polls):
            await asyncio.sleep(5)
            r = await client.get(
                f"{APIFY_API}/actor-runs/{run_id}",
                headers=headers,
            )
            r.raise_for_status()
            status = r.json().get("data", {}).get("status")

            if status == "SUCCEEDED":
                dataset_id = r.json().get("data", {}).get("defaultDatasetId")
                if not dataset_id:
                    raise RuntimeError("Apify: succeeded but no dataset found")

                # Fetch ALL results (not just 10)
                r2 = await client.get(
                    f"{APIFY_API}/datasets/{dataset_id}/items?format=json",
                    headers=headers,
                )
                r2.raise_for_status()
                items = r2.json()

                if status_callback:
                    await status_callback(f"Scraping done. {len(items)} results. Processing...")

                return items

            elif status in ("FAILED", "ABORTED", "TIMED-OUT"):
                raise RuntimeError(f"Apify run {status.lower()}")

            # Progress update every 30s
            if status_callback and i > 0 and i % 6 == 0:
                elapsed = (i + 1) * 5
                await status_callback(f"Still scraping... ({elapsed}s elapsed)")

    raise RuntimeError("Apify run timed out after 5 minutes")


# ---------------------------------------------------------------------------
# Parse Apify results into clean rows
# ---------------------------------------------------------------------------

def parse_results(items: list[dict]) -> list[list[str]]:
    """Convert raw Apify items into clean sheet rows with fixed columns.

    This is DETERMINISTIC — no LLM involved. Each field is extracted
    from known JSON paths in the Apify Google Maps output.
    """
    rows = []
    seen_names = set()  # dedup by name

    for item in items:
        name = (item.get("title") or item.get("name") or "").strip()
        if not name or name.lower() in seen_names:
            continue
        seen_names.add(name.lower())

        # Phone — try multiple paths
        phone = clean_phone(
            item.get("phone") or item.get("phoneUnformatted") or ""
        )

        # Email — try direct field first, then search in various text fields
        email = ""
        direct_email = item.get("email") or item.get("emails") or ""
        if isinstance(direct_email, list):
            valid = [e for e in direct_email if validate_email(e)]
            email = valid[0] if valid else ""
        elif isinstance(direct_email, str) and validate_email(direct_email):
            email = direct_email

        # If no direct email, try extracting from website or description
        if not email:
            text_sources = [
                item.get("website") or "",
                item.get("description") or "",
                json.dumps(item.get("additionalInfo") or {}),
            ]
            for src in text_sources:
                found = extract_emails_from_text(src)
                if found:
                    email = found[0]
                    break

        # Website
        website = (item.get("website") or item.get("url") or "").strip()
        # The google maps URL is different from the business website
        maps_url = (item.get("url") or item.get("googleMapsUrl") or "").strip()
        if website == maps_url:
            website = ""  # Don't duplicate — that's the maps link, not their site

        # Address
        address = (item.get("address") or item.get("street") or "").strip()
        city = (item.get("city") or item.get("neighborhood") or "").strip()

        # If address contains city, extract it
        if not city and address:
            parts = address.rsplit(",", 2)
            if len(parts) >= 2:
                city = parts[-2].strip() if len(parts) >= 3 else parts[-1].strip()

        # Rating & reviews
        rating = str(item.get("totalScore") or item.get("rating") or "")
        reviews = str(item.get("reviewsCount") or item.get("reviews") or "")

        # Category
        category = (item.get("categoryName") or item.get("category") or "").strip()

        row = [
            name,
            phone,
            email,
            website,
            address,
            city,
            rating,
            reviews,
            category,
            maps_url,
        ]
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Google Sheets writing (uses existing OAuth from OpenClaw)
# ---------------------------------------------------------------------------

SHEETS_API = "https://sheets.googleapis.com/v4/spreadsheets"


async def get_sheets_token(creds_dir: str) -> str:
    """Load OAuth token from OpenClaw's token.pickle."""
    import pickle
    token_path = Path(creds_dir) / "token.pickle"
    creds_path = Path(creds_dir) / "credentials.json"

    if not token_path.exists():
        raise FileNotFoundError(
            f"No token.pickle at {token_path}. "
            "Run google_auth.py on the server first to create it."
        )

    with open(token_path, "rb") as f:
        creds = pickle.load(f)

    # Refresh if expired
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
    """Write lead rows to Google Sheets. Returns stats."""
    token = await get_sheets_token(creds_dir)
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. Check if sheet/tab exists, create if not
        r = await client.get(
            f"{SHEETS_API}/{spreadsheet_id}",
            headers=headers,
        )
        r.raise_for_status()
        sheet_data = r.json()
        sheet_titles = [s["properties"]["title"] for s in sheet_data.get("sheets", [])]

        if sheet_name not in sheet_titles:
            # Create the tab
            await client.post(
                f"{SHEETS_API}/{spreadsheet_id}:batchUpdate",
                headers=headers,
                json={
                    "requests": [{
                        "addSheet": {
                            "properties": {"title": sheet_name}
                        }
                    }]
                },
            )

        # 2. Clear existing data in the tab
        await client.post(
            f"{SHEETS_API}/{spreadsheet_id}/values/{sheet_name}:clear",
            headers=headers,
            json={},
        )

        # 3. Write header + all rows
        all_values = [COLUMNS] + rows

        r = await client.put(
            f"{SHEETS_API}/{spreadsheet_id}/values/{sheet_name}!A1",
            headers=headers,
            params={"valueInputOption": "USER_ENTERED"},
            json={"values": all_values},
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
    """Scrape websites from rows to find missing emails. Returns (updated_rows, found_count)."""
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

        # Skip if already has email or no website
        if email or not website:
            continue

        if status_callback and i % 10 == 0:
            await status_callback(
                f"Enriching... ({i}/{len(rows)} checked, {found} emails found)"
            )

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.post(
                    "https://api.firecrawl.dev/v1/scrape",
                    headers=headers,
                    json={"url": website, "formats": ["markdown"]},
                )
                if r.status_code != 200:
                    continue
                data = r.json()
                markdown = data.get("data", {}).get("markdown", "")
                emails = extract_emails_from_text(markdown)
                if emails:
                    row[2] = emails[0]
                    found += 1

            # Rate limit — be nice to Firecrawl
            await asyncio.sleep(1)

        except Exception as e:
            log.warning("Firecrawl enrichment failed for %s: %s", website, e)
            continue

    return rows, found


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------

# Default sheet ID — can be overridden per command
DEFAULT_SHEET_ID = "16m1SQsP76wSc7HXsjAfeEDocEbuUDZj8gCxbk6B_J3w"

# Google workspace creds directory on experiment server
GOOGLE_CREDS_DIR = str(
    Path(__file__).resolve().parent.parent / "google-workspace"
)


async def handle(command: str, args: str, secrets: dict, cloud_chat) -> str:
    """Handle /leads commands."""

    if not args:
        return (
            "**Lead Scraper** — Apify Google Maps → Google Sheets\n\n"
            "**Usage:**\n"
            '  `/leads "plumbers in Miami"` — scrape & write to default sheet\n'
            '  `/leads "restaurants in Nicosia" SHEET_ID` — use specific sheet\n'
            '  `/leads "query" SHEET_ID 200` — scrape up to 200 results\n'
            '  `/leads enrich SHEET_ID` — scrape websites to find missing emails\n'
            '  `/leads status` — show last scrape stats\n\n'
            "**Default sheet:** Lista001\n"
            f"**Columns:** {', '.join(COLUMNS)}"
        )

    # Parse subcommands
    args_lower = args.strip().lower()

    # --- ENRICH subcommand ---
    if args_lower.startswith("enrich"):
        return await _handle_enrich(args, secrets)

    # --- STATUS subcommand ---
    if args_lower == "status":
        return _get_status()

    # --- MAIN SCRAPE COMMAND ---
    return await _handle_scrape(args, secrets)


# Store last scrape stats
_last_scrape: dict = {}


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


async def _handle_scrape(args: str, secrets: dict) -> str:
    """Run the full pipeline: scrape → parse → validate → write to sheets."""
    api_key = secrets.get("apify_api_key", "")
    if not api_key:
        return "Apify API key not configured."

    # Parse: "query" [sheet_id] [max_results]
    # The query can be quoted or not
    parts = args.strip().split()
    query = ""
    sheet_id = DEFAULT_SHEET_ID
    max_results = 100

    # Handle quoted query
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
        # Unquoted — take everything as the query unless last parts look like IDs/numbers
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
        return "Please provide a search query. Example: `/leads \"plumbers in Miami\"`"

    start_time = time.time()

    try:
        # Step 1: Scrape with Apify
        items = await run_apify_scrape(query, api_key, max_results)

        if not items:
            return f"No results for: {query}"

        # Step 2: Parse & validate (DETERMINISTIC — no LLM)
        rows = parse_results(items)

        if not rows:
            return f"Scraped {len(items)} items but none had usable data."

        # Step 3: Enrich with Firecrawl if available
        firecrawl_key = secrets.get("firecrawl_api_key", "")
        enriched_count = 0
        if firecrawl_key:
            # Only enrich rows without emails (max 20 to stay fast)
            no_email = [r for r in rows if not r[2] and r[3]]  # no email but has website
            if no_email:
                to_enrich = no_email[:20]
                _, enriched_count = await enrich_with_firecrawl(
                    to_enrich, firecrawl_key
                )

        # Step 4: Write to Google Sheets
        try:
            stats = await write_to_sheets(
                rows, sheet_id, GOOGLE_CREDS_DIR,
                sheet_name=query[:30].replace('"', '').strip(),
            )
            sheet_status = f"Written to sheet: {stats['rows_written']} rows"
        except Exception as e:
            log.error("Google Sheets write failed: %s", e)
            sheet_status = f"Sheet write FAILED: {e}"

        # Step 5: Calculate stats
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

        # Build report
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
            report += f"  Enriched via Firecrawl: +{enriched_count} emails\n"

        report += (
            f"\n**Sheet:** {sheet_status}\n"
            f"**Time:** {duration}s"
        )

        return report

    except Exception as e:
        log.error("Lead scrape pipeline failed: %s", e)
        return f"Pipeline error: {e}"


async def _handle_enrich(args: str, secrets: dict) -> str:
    """Enrich existing sheet data with emails from Firecrawl."""
    firecrawl_key = secrets.get("firecrawl_api_key", "")
    if not firecrawl_key:
        return "Firecrawl API key not configured. Add it with `/api set firecrawl_api_key KEY`"

    parts = args.strip().split()
    sheet_id = parts[1] if len(parts) > 1 else DEFAULT_SHEET_ID
    sheet_name = parts[2] if len(parts) > 2 else "Leads"

    try:
        # Read current sheet data
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

        # Skip header row
        rows = values[1:]

        # Pad rows to 10 columns
        rows = [r + [""] * (10 - len(r)) for r in rows]

        rows, found = await enrich_with_firecrawl(rows, firecrawl_key)

        if found:
            # Write back
            await write_to_sheets(rows, sheet_id, GOOGLE_CREDS_DIR, sheet_name)
            return f"Enrichment complete. Found {found} new emails from {len(rows)} websites."
        else:
            return "No new emails found via website scraping."

    except Exception as e:
        return f"Enrichment error: {e}"


def _pct(n: int, total: int) -> str:
    if total == 0:
        return "0%"
    return f"{round(n / total * 100)}%"
