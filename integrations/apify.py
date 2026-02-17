"""Apify integration — run actors for web scraping."""

import asyncio

import httpx

NAME = "apify"
COMMANDS = {
    "apify": "Run an Apify actor (scraping, data extraction)",
}
REQUIRED_SECRETS = ["apify_api_key"]

APIFY_API = "https://api.apify.com/v2"

# Named actors — shortcuts for common scrapers
NAMED_ACTORS = {
    "linkedin": "Y6wwxi8fkYNRe7MC4",
    "instagram": "9kFt1A08XFoM4Am9r",
    "web": "apify/website-content-crawler",
    "google": "apify/google-search-scraper",
}


async def handle(command: str, args: str, secrets: dict, cloud_chat) -> str:
    api_key = secrets.get("apify_api_key", "")
    if not api_key:
        return "Apify API key not configured in secrets.json."

    if not args:
        actors = "\n".join(f"  `{k}` — {v}" for k, v in NAMED_ACTORS.items())
        return (
            "Usage: `/apify <actor> <input>`\n\n"
            "**Named actors:**\n" + actors + "\n\n"
            "**Examples:**\n"
            "  `/apify web https://example.com`\n"
            "  `/apify google \"best raspberry pi projects\"`\n"
            "  `/apify apify/some-actor {\"key\": \"value\"}`"
        )

    parts = args.split(maxsplit=1)
    actor_name = parts[0]
    actor_input_raw = parts[1] if len(parts) > 1 else ""

    # Resolve named actors
    actor_id = NAMED_ACTORS.get(actor_name, actor_name)

    # Parse input — could be JSON or a simple URL/query
    actor_input = _parse_input(actor_name, actor_input_raw)

    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        # Start actor run
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                f"{APIFY_API}/acts/{actor_id}/runs",
                headers=headers,
                json=actor_input,
            )
            r.raise_for_status()
            run_data = r.json().get("data", {})
            run_id = run_data.get("id")

            if not run_id:
                return "Actor started but no run ID returned."

        # Poll for completion (max 120s)
        async with httpx.AsyncClient(timeout=30.0) as client:
            for _ in range(24):
                await asyncio.sleep(5)
                r = await client.get(
                    f"{APIFY_API}/actor-runs/{run_id}",
                    headers=headers,
                )
                r.raise_for_status()
                status = r.json().get("data", {}).get("status")

                if status == "SUCCEEDED":
                    # Fetch results
                    dataset_id = r.json().get("data", {}).get("defaultDatasetId")
                    if not dataset_id:
                        return "Actor succeeded but no dataset found."

                    r2 = await client.get(
                        f"{APIFY_API}/datasets/{dataset_id}/items?limit=10&format=json",
                        headers=headers,
                    )
                    r2.raise_for_status()
                    items = r2.json()

                    if not items:
                        return "Actor completed but returned no results."

                    # Format results
                    result = _format_results(items)

                    # Optionally summarize with cloud model
                    if len(result) > 2000:
                        summary = await cloud_chat([
                            {"role": "system", "content": "Summarize this scraped data concisely. Keep key facts and data points."},
                            {"role": "user", "content": result[:8000]},
                        ])
                        return f"**Apify results** ({len(items)} items):\n\n{summary}"

                    return f"**Apify results** ({len(items)} items):\n\n{result}"

                elif status in ("FAILED", "ABORTED", "TIMED-OUT"):
                    return f"Actor run {status.lower()}."

        return "Actor timed out after 120s. Check Apify dashboard for results."

    except Exception as e:
        return f"Apify error: {e}"


def _parse_input(actor_name: str, raw: str) -> dict:
    """Parse actor input — smart defaults for named actors."""
    import json

    # Try JSON first
    if raw.startswith("{"):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

    # Smart defaults for named actors
    if actor_name == "web":
        return {"startUrls": [{"url": raw}], "maxCrawlPages": 5}
    elif actor_name == "google":
        return {"queries": raw, "maxPagesPerQuery": 1}
    elif actor_name == "linkedin":
        return {"profileUrls": [raw]}
    elif actor_name == "instagram":
        return {"usernames": [raw.lstrip("@")]}

    # Fallback — treat as URL or query
    if raw.startswith("http"):
        return {"startUrls": [{"url": raw}]}

    return {"input": raw}


def _format_results(items: list) -> str:
    """Format Apify dataset items into readable text."""
    import json

    results = []
    for i, item in enumerate(items[:10], 1):
        # Try to extract common fields
        title = item.get("title") or item.get("name") or item.get("headline") or ""
        url = item.get("url") or item.get("link") or ""
        text = item.get("text") or item.get("description") or item.get("body") or ""

        if title or url or text:
            entry = f"**{i}.** "
            if title:
                entry += f"{title}\n"
            if url:
                entry += f"{url}\n"
            if text:
                entry += text[:500] + "\n"
            results.append(entry)
        else:
            # Fallback: dump as JSON
            results.append(f"**{i}.** ```{json.dumps(item, indent=2)[:500]}```")

    return "\n".join(results)
