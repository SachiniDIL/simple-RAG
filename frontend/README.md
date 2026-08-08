# Simple RAG — frontend

A single-page Next.js (App Router) UI for querying the Simple RAG backend. It
lets you ask a question, optionally enable hybrid search and answer
generation, and view the retrieved chunks (and generated answer, if
requested).

## How it connects to the backend

This app is a plain client that calls the FastAPI backend's `POST /query`
endpoint over HTTP — it does not talk to the vector store, LLM, or any other
backend internals directly. The request is fired with TanStack Query's
`useMutation` from [app/page.tsx](app/page.tsx).

The backend base URL is read from `NEXT_PUBLIC_API_URL`, defaulting to
`http://127.0.0.1:8000` if unset. Copy the example env file and adjust it if
your backend runs elsewhere:

```bash
cp .env.local.example .env.local
```

## Running both servers for local dev

This frontend is a separate project from the Python backend (which lives in
the parent directory). You need both running at the same time:

**Backend** (from the parent project directory, in a Python environment with
the project's dependencies installed):

```bash
uvicorn api:app --reload
```

This serves the API at `http://127.0.0.1:8000`.

**Frontend** (from this `frontend/` directory):

```bash
npm run dev
```

This serves the UI at `http://localhost:3000`. Open it in your browser, ask a
question, and it will call the backend above.

If the backend isn't running, submitting a query shows a clear "couldn't
reach the backend" error rather than failing silently.
