# Google Sheets & Docs Integration Setup

## Overview

Berryclaw integrates with Google Sheets and Google Docs via OAuth2. Users authorize through Telegram — no server-side browser needed.

**Commands available after setup:**
- `/sheets` — Read, write, append to Google Sheets
- `/docs` — Read, append, ask questions about Google Docs
- `/leads` — Automated Apify Google Maps scraper → Google Sheets pipeline

## Prerequisites

### 1. Create a Google Cloud Project

1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Create a new project (or use an existing one)
3. Note the **Project Number** (you'll need it later)

### 2. Enable Required APIs

Go to **APIs & Services > Library** and enable:

- **Google Sheets API** (required)
- **Google Drive API** (required for `/sheets list`)
- **Google Docs API** (required for `/docs`)

Direct links (replace `PROJECT_NUMBER` with yours):
```
https://console.developers.google.com/apis/api/sheets.googleapis.com/overview?project=PROJECT_NUMBER
https://console.developers.google.com/apis/api/drive.googleapis.com/overview?project=PROJECT_NUMBER
https://console.developers.google.com/apis/api/docs.googleapis.com/overview?project=PROJECT_NUMBER
```

### 3. Configure OAuth Consent Screen

1. Go to **APIs & Services > OAuth consent screen**
2. Choose **External** user type
3. Fill in the required fields (app name, support email)
4. Add scopes:
   - `https://www.googleapis.com/auth/spreadsheets`
   - `https://www.googleapis.com/auth/documents`
   - `https://www.googleapis.com/auth/drive`
5. Add **Test users** — add the Google email address that will authorize the bot
6. Save

> **Important:** While the app is in "Testing" mode, only emails listed as test users can authorize. You can publish the app later to remove this restriction.

### 4. Create OAuth Credentials

1. Go to **APIs & Services > Credentials**
2. Click **Create Credentials > OAuth client ID**
3. Application type: **Desktop app** (or "Installed application")
4. Name it anything (e.g., "Berryclaw")
5. Click **Create**
6. Download the JSON file
7. Rename it to `credentials.json`

### 5. Place credentials.json

Put `credentials.json` in the `google-workspace/` directory next to `berryclaw.py`:

```
berryclaw/
  berryclaw.py
  google-workspace/
    credentials.json    <-- put it here
    token.pickle        <-- created automatically after auth
```

## Authorization Flow

1. In Telegram, type `/sheets`
2. Bot shows "Authorize Google Account" button
3. Tap it — bot sends an authorization URL
4. Copy the URL and open it in your browser
5. Sign in with your Google account (must be a test user if app is in Testing mode)
6. Allow the requested permissions
7. Google shows you an authorization code
8. Paste the code back in Telegram as your next message
9. Bot exchanges the code for a token and saves it

After this, `/sheets` and `/docs` work. The token auto-refreshes — you only need to authorize once.

## Usage

### Google Sheets

```
/sheets                              — Show action buttons
/sheets read <sheet_id> [range]      — Read data (default: A1:Z100)
/sheets write <sheet_id> <range> <value>  — Write to a cell
/sheets append <sheet_id> <range> val1 | val2 | val3  — Append a row
/sheets list                         — List recent spreadsheets (needs Drive API)
```

**Finding your Sheet ID:** Open the spreadsheet in your browser. The ID is in the URL:
```
https://docs.google.com/spreadsheets/d/SHEET_ID_HERE/edit
```

### Google Docs

```
/docs read <doc_id>                  — Read document content
/docs append <doc_id> <text>         — Append text to document
/docs ask <doc_id> <question>        — Ask AI a question about the document
```

### Lead Scraper (Apify + Sheets)

```
/leads "plumbers in Miami"           — Scrape Google Maps → write to default sheet
/leads "query" SHEET_ID              — Scrape and write to specific sheet
/leads enrich SHEET_ID               — Scrape websites for missing emails
/leads status                        — Check scraper status
```

The lead scraper uses a **deterministic pipeline** — no LLM touches the data:
1. Apify Google Maps scraper runs
2. Results are parsed with fixed field extraction
3. Emails validated with regex, junk filtered
4. Phones normalized
5. Deduplication by business name
6. Written to Google Sheets with consistent columns

## Troubleshooting

| Error | Fix |
|-------|-----|
| "Not authorized yet" | Run `/sheets` and complete the auth flow |
| "Token expired or invalid" | Click "Re-authorize" and repeat the auth flow |
| "403 Forbidden" on list | Enable Google Drive API on your Cloud project |
| "access_denied" during auth | Add your email as a test user in OAuth consent screen |
| "400 invalid_request" during auth | Copy the URL to a desktop browser (some mobile browsers mangle it) |

## Security Notes

- `credentials.json` contains your OAuth client ID/secret — do NOT commit it to git
- `token.pickle` contains your access/refresh tokens — do NOT commit it to git
- Both files are in `.gitignore` by default
- Tokens are stored locally on the server only
- The bot never sends your Google credentials over Telegram
