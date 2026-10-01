import asyncio
import logging
import os
import re
import shutil
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from rag_agent.agent import root_agent
from rag_agent.ingest import (
    SHARED_DIR,
    USERS_DIR,
    delete_document_chunks,
    delete_owner_chunks,
    ingest_pdf,
    pdfs_in,
    user_dir,
    visible_documents,
)

load_dotenv()
log = logging.getLogger("docchat")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.join(APP_DIR, "demo_docs")   # sample PDFs shipped with the app


def env_flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


APP_NAME = "doc_chat"
MAX_UPLOAD_MB = 10                 # reject bigger PDFs
MAX_QUESTION_CHARS = 1000          # reject huge prompts
AGENT_TIMEOUT_S = 60               # give up if the agent hangs
RATE_LIMIT = 20                    # requests allowed ...
RATE_WINDOW_S = 60                 # ... per this many seconds, per IP address
ALLOW_UPLOADS = env_flag("ALLOW_UPLOADS", "true")
DAILY_ASK_LIMIT = int(os.getenv("DAILY_ASK_LIMIT", "300"))       # questions per day for the whole app; 0 = no cap
DAILY_UPLOAD_LIMIT = int(os.getenv("DAILY_UPLOAD_LIMIT", "60"))  # uploads per day for the whole app; 0 = no cap
MAX_DOCS_PER_VISITOR = int(os.getenv("MAX_DOCS_PER_VISITOR", "5"))
VISITOR_TTL_HOURS = float(os.getenv("VISITOR_TTL_HOURS", "24"))  # private uploads are deleted after this long
SEED_ATTEMPTS = 3                  # tries per demo document at start-up
SEED_RETRY_DELAY_S = 5             # waits 5s, then 10s, between tries


# ---------- who is asking? ----------
# The page creates a random ID once per browser and sends it with every request. It works like an
# unlisted link: nobody can reach another visitor's documents without knowing their (random) ID.
# It is a privacy boundary for a demo, not a login system.
VISITOR_RE = re.compile(r"^[A-Za-z0-9_-]{16,64}$")


def get_visitor(request: Request) -> str:
    visitor = request.headers.get("x-visitor-id", "")
    if not VISITOR_RE.match(visitor):
        raise HTTPException(400, "Missing or invalid visitor ID. Reload the page and try again.")
    return visitor


# ---------- guardrail: rate limit per IP address ----------
_hits = defaultdict(deque)


def rate_limit(request: Request):
    ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    recent = _hits[ip]
    while recent and now - recent[0] > RATE_WINDOW_S:
        recent.popleft()
    if len(recent) >= RATE_LIMIT:
        raise HTTPException(429, "Too many requests. Wait a minute and try again.")
    recent.append(now)


# ---------- guardrail: daily caps for the whole app (protect your Gemini quota) ----------
_daily = {"day": None, "asks": 0, "uploads": 0}


def check_daily_limit(kind: str):
    today = time.strftime("%Y-%m-%d", time.gmtime())
    if _daily["day"] != today:
        _daily.update(day=today, asks=0, uploads=0)
    limit = DAILY_ASK_LIMIT if kind == "asks" else DAILY_UPLOAD_LIMIT
    if limit and _daily[kind] >= limit:
        what = "question" if kind == "asks" else "upload"
        raise HTTPException(429, f"This demo has reached its daily {what} limit. Please try again tomorrow.")
    _daily[kind] += 1


# Passages retrieved earlier in each chat. If the agent answers a follow-up from memory without
# searching again, its citations can still be matched to a passage to highlight.
_session_passages = {}            # {(visitor, session_id): [passages]}
MAX_PASSAGES_PER_SESSION = 40


# ---------- background jobs ----------
def seed_demo_documents():
    """Index the PDFs in demo_docs/ as shared sample documents (skips ones already loaded).

    A host with a temporary disk forgets everything on restart, so these are loaded again at every
    start-up. A file only appears once it is indexed. A first request to Gemini can fail briefly
    (rate limit, network), so each file is retried.
    """
    if not os.path.isdir(DEMO_DIR):
        return
    os.makedirs(SHARED_DIR, exist_ok=True)
    for name in sorted(os.listdir(DEMO_DIR)):
        target = os.path.join(SHARED_DIR, name)
        if not name.lower().endswith(".pdf") or os.path.exists(target):
            continue
        source = os.path.join(DEMO_DIR, name)
        for attempt in range(1, SEED_ATTEMPTS + 1):
            try:
                count = ingest_pdf(source)
                shutil.copyfile(source, target)
                log.info("Loaded demo document %s (%d chunks)", name, count)
                break
            except Exception:
                log.exception("Could not load demo document %s (attempt %d of %d)", name, attempt, SEED_ATTEMPTS)
                if attempt < SEED_ATTEMPTS:
                    time.sleep(SEED_RETRY_DELAY_S * attempt)


def purge_old_visitors(now=None):
    """Delete private folders (and their indexed text) not touched for VISITOR_TTL_HOURS."""
    if not os.path.isdir(USERS_DIR):
        return 0
    cutoff = (now if now is not None else time.time()) - VISITOR_TTL_HOURS * 3600
    removed = 0
    for visitor in os.listdir(USERS_DIR):
        folder = os.path.join(USERS_DIR, visitor)
        if not os.path.isdir(folder):
            continue
        stamps = [os.path.getmtime(os.path.join(folder, f)) for f in os.listdir(folder)] or [os.path.getmtime(folder)]
        if max(stamps) >= cutoff:
            continue
        try:
            delete_owner_chunks(visitor)
            shutil.rmtree(folder, ignore_errors=True)
            for key in [k for k in _session_passages if k[0] == visitor]:
                del _session_passages[key]
            removed += 1
        except Exception:
            log.exception("Could not clean up visitor data")
    return removed


async def purge_loop():
    while True:
        try:
            await asyncio.to_thread(purge_old_visitors)
        except Exception:
            log.exception("Cleanup failed")
        await asyncio.sleep(3600)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Both run in the background so the server starts answering requests immediately.
    app.state.seed_task = asyncio.create_task(asyncio.to_thread(seed_demo_documents))
    app.state.purge_task = asyncio.create_task(purge_loop())
    yield
    app.state.purge_task.cancel()


app = FastAPI(lifespan=lifespan)
sessions = InMemorySessionService()
runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=sessions)
os.makedirs(SHARED_DIR, exist_ok=True)
os.makedirs(USERS_DIR, exist_ok=True)


# ---------- helpers ----------
class Question(BaseModel):
    question: str
    session_id: str | None = None


def safe_name(filename: str) -> str:
    """Keep only the file name, and replace odd characters."""
    name = os.path.basename(filename or "")
    return re.sub(r"[^\w.\- ]", "_", name).strip()


async def save_with_limit(file: UploadFile, path: str) -> bool:
    """Save the upload in 1 MB pieces. Returns False if it grew past the limit."""
    limit = MAX_UPLOAD_MB * 1024 * 1024
    size, too_big = 0, False
    with open(path, "wb") as out:
        while True:
            piece = await file.read(1024 * 1024)
            if not piece:
                break
            size += len(piece)
            if size > limit:
                too_big = True
                break
            out.write(piece)
    if too_big:
        os.remove(path)
    return not too_big


def merge_sources(new, old):
    """This turn's passages first, then earlier ones, without duplicates."""
    merged, seen = [], set()
    for p in list(new) + list(old):
        key = (p["source"], p["page"], p["text"][:60])
        if key not in seen:
            seen.add(key)
            merged.append(p)
    return merged[:MAX_PASSAGES_PER_SESSION]


def remove_quietly(path: str):
    try:
        os.remove(path)
    except OSError:
        pass


# ---------- endpoints ----------
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/documents")
async def documents(visitor: str = Depends(get_visitor)):
    return {
        "documents": visible_documents(visitor),       # sample documents + this visitor's own
        "mine": pdfs_in(user_dir(visitor)),            # the ones this visitor may remove
        "uploads_enabled": ALLOW_UPLOADS,
        "max_docs": MAX_DOCS_PER_VISITOR,
        "ttl_hours": VISITOR_TTL_HOURS,
    }


@app.post("/upload", dependencies=[Depends(rate_limit)])
async def upload(file: UploadFile, visitor: str = Depends(get_visitor)):
    if not ALLOW_UPLOADS:
        raise HTTPException(403, "Uploads are turned off on this demo. Ask about the sample documents instead.")
    name = safe_name(file.filename)
    if not name.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are allowed.")
    if name in pdfs_in(SHARED_DIR):
        raise HTTPException(409, "A sample document already uses this name. Rename your file and try again.")

    mine_dir = user_dir(visitor)
    mine = pdfs_in(mine_dir)
    if name not in mine and len(mine) >= MAX_DOCS_PER_VISITOR:
        raise HTTPException(409, f"You can keep up to {MAX_DOCS_PER_VISITOR} documents. Remove one first.")
    check_daily_limit("uploads")

    os.makedirs(mine_dir, exist_ok=True)
    final_path = os.path.join(mine_dir, name)
    # Work on a temporary copy, so a failed upload can never damage a good file that is already there.
    temp_path = os.path.join(mine_dir, f"{name}.{uuid.uuid4().hex}.part")
    try:
        if not await save_with_limit(file, temp_path):
            raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB} MB. Upload a smaller PDF.")

        # A real PDF starts with these 5 bytes, whatever the file is called.
        with open(temp_path, "rb") as f:
            if f.read(5) != b"%PDF-":
                raise HTTPException(400, "This file is not a valid PDF.")

        try:
            # Indexing is slow (it calls Gemini). Run it in a worker thread, otherwise the whole
            # server, including /health, freezes until it finishes.
            count = await asyncio.to_thread(ingest_pdf, temp_path, name, visitor)
        except ValueError as e:             # no readable text, or too long
            raise HTTPException(422, str(e))
        except Exception:                   # corrupt file, embedding service busy, ...
            log.exception("Ingestion failed for %s", name)
            raise HTTPException(
                422,
                "Could not process this PDF. It may be corrupt or password-protected, "
                "or the embedding service may be busy. Try again in a moment.",
            )

        os.replace(temp_path, final_path)   # the file appears in the list only once it is fully indexed
    finally:
        remove_quietly(temp_path)           # does nothing if the file was already moved
    return {"file": name, "chunks": count}


@app.delete("/documents/{name}", dependencies=[Depends(rate_limit)])
async def delete_document(name: str, visitor: str = Depends(get_visitor)):
    name = safe_name(name)
    if name in pdfs_in(SHARED_DIR) and name not in pdfs_in(user_dir(visitor)):
        raise HTTPException(403, "Sample documents can't be removed.")
    path = os.path.join(user_dir(visitor), name)
    if not name.lower().endswith(".pdf") or not os.path.isfile(path):
        raise HTTPException(404, "Document not found.")
    await asyncio.to_thread(delete_document_chunks, name, visitor)
    remove_quietly(path)
    return {"deleted": name}


@app.get("/files/{name}")
async def get_file(name: str, visitor: str = Depends(get_visitor)):
    """A visitor can open their own PDFs and the shared samples, and nothing else."""
    name = safe_name(name)
    if name.lower().endswith(".pdf"):
        for folder in (user_dir(visitor), SHARED_DIR):
            path = os.path.join(folder, name)
            if os.path.isfile(path):
                return FileResponse(path, media_type="application/pdf", headers={"Cache-Control": "private, no-store"})
    raise HTTPException(404, "Document not found.")


async def run_agent(question: str, sid: str, visitor: str):
    # The visitor ID is the ADK user_id: sessions are separate per visitor, and the search tool
    # reads it to decide which documents may be searched.
    existing = await sessions.get_session(app_name=APP_NAME, user_id=visitor, session_id=sid)
    if not existing:
        await sessions.create_session(app_name=APP_NAME, user_id=visitor, session_id=sid)

    message = types.Content(role="user", parts=[types.Part(text=question)])
    answer, passages = "", []
    async for event in runner.run_async(user_id=visitor, session_id=sid, new_message=message):
        for fr in (event.get_function_responses() or []):
            if fr.name == "search_documents":
                passages.extend((fr.response or {}).get("passages", []))
        if event.is_final_response() and event.content and event.content.parts:
            answer = event.content.parts[0].text or ""
    return answer, passages


@app.post("/ask", dependencies=[Depends(rate_limit)])
async def ask(q: Question, visitor: str = Depends(get_visitor)):
    question = q.question.strip()
    if not question:
        raise HTTPException(400, "Type a question first.")
    if len(question) > MAX_QUESTION_CHARS:
        raise HTTPException(400, f"Question is too long. Keep it under {MAX_QUESTION_CHARS} characters.")

    check_daily_limit("asks")
    sid = q.session_id or str(uuid.uuid4())
    try:
        answer, passages = await asyncio.wait_for(run_agent(question, sid, visitor), timeout=AGENT_TIMEOUT_S)
    except asyncio.TimeoutError:
        raise HTTPException(504, "The agent took too long. Try a simpler question.")
    except Exception:
        log.exception("Agent failed")
        raise HTTPException(502, "The agent hit an error. Try again in a moment.")

    key = (visitor, sid)
    sources = merge_sources(passages, _session_passages.get(key, []))
    _session_passages[key] = sources

    return {
        "answer": answer or "Sorry, I couldn't produce an answer.",
        "session_id": sid,
        "sources": sources,
    }


# The page itself. Registered last so it cannot shadow the routes above.
app.mount("/", StaticFiles(directory="static", html=True), name="static")
