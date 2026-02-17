# Berryclaw Build Environment

You are running on a Raspberry Pi 5 via Telegram. The user is chatting with you from their phone. Keep responses concise — they're reading on a small screen.

## Project Rules

Every new project gets its own folder:

```
~/projects/<project-name>/
```

Before creating a project, check what already exists: `ls ~/projects/`

Never create files outside `~/projects/`. Never modify `~/raspberryclaw/` (that's the Telegram bot).

## Port Allocation

Use ports **3000-3099** for web servers. Before binding a port, check if it's free:

```bash
lsof -i :<port>
```

Start from 3000 and go up. If 3000 is taken, use 3001, etc.

**Ports already in use (DO NOT touch):**
- 22 — SSH
- 11434 — Ollama
- 631 — CUPS

## URLs & Access

When you start a web server, the user can access it at:

```
http://10.10.49.41:<port>
```

Always tell the user the full URL after starting a server. Example:
> "Your app is running at http://10.10.49.41:3000"

The Pi is on a university LAN. This URL works from any device on the same network.

## Environment

- **Hardware**: Raspberry Pi 5, 8GB RAM, ARM64 (aarch64), no GPU
- **OS**: Debian/Raspberry Pi OS (bookworm)
- **Python**: 3.13+ (`python3`)
- **Node.js**: 22+ (`node`, `npm`)
- **Disk**: ~95GB free on `/`
- **Ollama**: Running on localhost:11434 (for AI features if needed)

## Authentication — MANDATORY

**Every web project MUST have basic auth.** The Pi is on a university LAN — anyone on the network can access your ports. No exceptions.

Credentials are stored in `~/projects/.auth` (JSON):
```json
{"username": "admin", "password": "berryclaw"}
```

Read this file at startup and require login before showing any content.

### Python (FastAPI)

```python
import json
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pathlib import Path

security = HTTPBasic()
_auth = json.loads((Path.home() / "projects" / ".auth").read_text())

def verify(creds: HTTPBasicCredentials = Depends(security)):
    if creds.username != _auth["username"] or creds.password != _auth["password"]:
        raise HTTPException(status_code=401, detail="Unauthorized",
                            headers={"WWW-Authenticate": "Basic"})
    return creds.username

# Add Depends(verify) to every route:
# @app.get("/", dependencies=[Depends(verify)])
```

### Python (Flask / simple server)

```python
import json, functools
from flask import request, Response
from pathlib import Path

_auth = json.loads((Path.home() / "projects" / ".auth").read_text())

def require_auth(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or auth.username != _auth["username"] or auth.password != _auth["password"]:
            return Response("Login required", 401, {"WWW-Authenticate": "Basic realm='Login'"})
        return f(*args, **kwargs)
    return decorated
```

### Node.js (Express)

```javascript
const fs = require('fs');
const path = require('path');
const auth = JSON.parse(fs.readFileSync(path.join(require('os').homedir(), 'projects', '.auth')));

app.use((req, res, next) => {
  const b64 = (req.headers.authorization || '').split(' ')[1] || '';
  const [user, pass] = Buffer.from(b64, 'base64').toString().split(':');
  if (user === auth.username && pass === auth.password) return next();
  res.set('WWW-Authenticate', 'Basic realm="Login"');
  res.status(401).send('Login required');
});
```

### Static HTML (use this server instead of python3 -m http.server)

```python
#!/usr/bin/env python3
"""Authenticated static file server. Usage: python3 serve.py [port]"""
import json, http.server, base64, sys
from pathlib import Path

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
_auth = json.loads((Path.home() / "projects" / ".auth").read_text())

class AuthHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        auth_header = self.headers.get("Authorization", "")
        if auth_header.startswith("Basic "):
            decoded = base64.b64decode(auth_header[6:]).decode()
            user, pw = decoded.split(":", 1)
            if user == _auth["username"] and pw == _auth["password"]:
                return super().do_GET()
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="Login"')
        self.end_headers()

http.server.HTTPServer(("0.0.0.0", PORT), AuthHandler).serve_forever()
```

**NEVER use `python3 -m http.server` directly** — it has no auth. Use the script above for static files.

## What to Use for Web Projects

For quick prototypes, prefer:
- **Python**: FastAPI/Flask for APIs, the authenticated static server (above) for HTML files
- **Node.js**: Vite, Next.js, or Express
- **Static HTML**: Use the authenticated `serve.py` above — never bare `python3 -m http.server`

Always install dependencies locally in the project folder (use venvs for Python, `npm install` for Node).

## Running Servers — CRITICAL

Servers MUST survive after you (Claude Code) exit. **Always use `setsid`** to start servers in their own process group:

```bash
setsid python3 serve.py 3000 > server.log 2>&1 &
```

**NEVER start servers without `setsid`.** Without it, the server dies when the tmux session ends.

For Node.js:
```bash
setsid node server.js > server.log 2>&1 &
setsid npx vite --host 0.0.0.0 --port 3000 > server.log 2>&1 &
```

After starting:
1. Verify it's running: `lsof -i :<port>`
2. Tell the user the full URL: `http://10.10.49.41:<port>`
3. To stop later: `kill $(lsof -t -i :<port>)`

## Constraints

- **No GPU** — don't try CUDA, use CPU-only libraries
- **ARM64** — most packages work, but check if something requires x86
- **8GB RAM** — be mindful with large dependencies. Ollama is always running (~200MB + model)
- **No root** — you run as user `franciscoandsam`. Use `sudo` only if truly needed
- **No Docker** — not installed, don't suggest it

## Communication Style

The user reads your output on Telegram (small screen). Be brief:
- Don't dump huge code blocks — write files instead
- Summarize what you did in 2-3 lines
- Always end with the URL or next step
