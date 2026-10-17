# gemini2api

An API proxy that exposes Gemini Web through OpenAI compatible and native Gemini interfaces.

## Quick start

Requirements: Python 3.11+ and uv.

Install dependencies:

```bash
uv sync
```

Create a `.env` file and set `API_KEY` to a secret of your own:

```dotenv
API_KEY=choose-a-long-random-secret
```

Start the service:

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8080
```
