# Agent Rules

## Data integrity
- Never make up data. If Apify returns no email, the email field stays blank.
- Always validate emails with regex before writing to sheets.
- Dedup results by business name.

## Pipeline order
When asked to scrape leads:
1. Confirm the query and target sheet
2. Run Apify Google Maps scraper
3. Parse results (deterministic, no LLM)
4. Validate emails/phones
5. Write to Google Sheets
6. Report stats: total, with email, with phone, with website

## Error handling
- If Apify fails, report immediately. Don't retry silently.
- If Google Sheets auth expires, tell the user to refresh token.pickle.
- If Firecrawl enrichment finds 0 emails, report that too.
