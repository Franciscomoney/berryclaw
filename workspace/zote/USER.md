# User Context — Zote Group

## Who Uses This
- Lead generation team requesting business scrapes from Google Maps
- Data goes into Google Sheets for the sales/outreach team
- Priority: accurate emails and phone numbers
- Quality over quantity — clean data matters more than volume

## Key Resources
- Default lead sheet: Lista001 (`16m1SQsP76wSc7HXsjAfeEDocEbuUDZj8gCxbk6B_J3w`)
- Google Sheets and Docs are connected via OAuth2 (token saved locally)
- Apify runs the Google Maps scraper (actor: compass/crawler-google-places)
- Firecrawl handles website scraping for email enrichment

## What Users Typically Ask
- "Find me plumbers in Miami" → use `/leads "plumbers in Miami"`
- "Check the sheet" → use `/sheets read SHEET_ID`
- "How many leads have emails?" → read sheet, count non-empty email column
- "Enrich the missing emails" → use `/leads enrich SHEET_ID`
- "Scrape this website" → use `/scrape URL`
- "Read this Google Doc" → use `/docs read DOC_ID`

## What Users Do NOT Want
- Fabricated data — if there is no email, leave it blank
- Long explanations — give numbers and results first
- Unverified claims — always check before reporting
