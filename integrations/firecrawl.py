"""Firecrawl integration — scrape and crawl websites."""

import httpx

NAME = "firecrawl"
COMMANDS = {
    "scrape": "Scrape a URL and return clean markdown",
    "crawl": "Crawl a site and extract pages (returns first page)",
}
REQUIRED_SECRETS = ["firecrawl_api_key"]

FIRECRAWL_API = "https://api.firecrawl.dev/v1"


async def handle(command: str, args: str, secrets: dict, cloud_chat) -> str:
    api_key = secrets.get("firecrawl_api_key", "")
    if not api_key:
        return "Firecrawl API key not configured in secrets.json."

    if not args:
        return f"Usage: `/{command} <url> [question]`\n\nExample: `/{command} https://example.com`"

    parts = args.split(maxsplit=1)
    url = parts[0]
    question = parts[1] if len(parts) > 1 else ""

    if not url.startswith("http"):
        url = "https://" + url

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        if command == "scrape":
            result = await _scrape(url, headers)
        elif command == "crawl":
            result = await _crawl(url, headers)
        else:
            return f"Unknown command: {command}"
    except Exception as e:
        return f"Firecrawl error: {e}"

    if not result:
        return "No content extracted from that URL."

    # If user asked a question, send scraped content to cloud model
    if question:
        answer = await cloud_chat([
            {"role": "system", "content": "Answer the user's question based on the following web content. Be concise."},
            {"role": "user", "content": f"Content from {url}:\n\n{result[:8000]}\n\n---\n\nQuestion: {question}"},
        ])
        return f"**Source:** {url}\n\n{answer}"

    # Truncate for Telegram
    if len(result) > 3500:
        result = result[:3500] + "\n\n... (truncated)"

    return f"**Scraped:** {url}\n\n{result}"


async def _scrape(url: str, headers: dict) -> str:
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(
            f"{FIRECRAWL_API}/scrape",
            headers=headers,
            json={"url": url, "formats": ["markdown"]},
        )
        r.raise_for_status()
        data = r.json()

        if not data.get("success"):
            return f"Scrape failed: {data.get('error', 'unknown error')}"

        return data.get("data", {}).get("markdown", "")


async def _crawl(url: str, headers: dict) -> str:
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Start crawl
        r = await client.post(
            f"{FIRECRAWL_API}/crawl",
            headers=headers,
            json={"url": url, "limit": 5},
        )
        r.raise_for_status()
        data = r.json()

        if not data.get("success"):
            return f"Crawl failed: {data.get('error', 'unknown error')}"

        crawl_id = data.get("id")
        if not crawl_id:
            return "Crawl started but no ID returned."

        # Poll for results (max 60s)
        import asyncio
        for _ in range(12):
            await asyncio.sleep(5)
            r = await client.get(
                f"{FIRECRAWL_API}/crawl/{crawl_id}",
                headers=headers,
            )
            r.raise_for_status()
            status = r.json()

            if status.get("status") == "completed":
                pages = status.get("data", [])
                if not pages:
                    return "Crawl completed but no pages found."

                results = []
                for page in pages[:5]:
                    title = page.get("metadata", {}).get("title", "Untitled")
                    md = page.get("markdown", "")[:1000]
                    results.append(f"### {title}\n{md}")

                return "\n\n---\n\n".join(results)

        return "Crawl timed out after 60s. Try `/scrape` for a single page."
