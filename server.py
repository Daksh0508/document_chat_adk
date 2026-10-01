import asyncio
import logging
import os
import re
import time
import shutil
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from rag_agent.agent import root_agent
from rag_agent.ingest import ingest_pdf, UPLOAD_DIR

load_dotenv()
log = logging.getLogger("docchat")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.join(APP_DIR, "demo_docs")   # sample PDFs shipped with the app


def env_flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


ALLOW_UPLOADS = env_flag("ALLOW_UPLOADS", "true")        # set to false on a public demo
DAILY_ASK_LIMIT = int(os.getenv("DAILY_ASK_LIMIT", "300"))  # questions per day for the whole app; 0 = no cap


SEED_ATTEMPTS = 3            # tries per demo document at start-up
SEED_RETRY_DELAY_S = 5       # waits 5s, then 10s, between tries


def seed_demo_documents():
    """Index the PDFs in demo_docs/ and make them available (skips ones already loaded).

    A host with a temporary disk forgets everything on restart, so the demo documents
    are loaded again at every start-up. A file only appears in uploads/ once it is indexed.
    A first request to Gemini can fail briefly (rate limit, network), so each file is retried.
    """
    if not os.path.isdir(DEMO_DIR):
        return
    for name in sorted(os.listdir(DEMO_DIR)):
        target = os.path.join(UPLOAD_DIR, name)
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Run in the background so the server starts answering requests immediately.
    app.state.seed_task = asyncio.create_task(asyncio.to_thread(seed_demo_documents))
    yield


app = FastAPI(lifespan=lifespan)

APP_NAME = "doc_chat"
MAX_UPLOAD_MB = 10          # reject bigger PDFs
MAX_QUESTION_CHARS = 1000   # reject huge prompts
AGENT_TIMEOUT_S = 60        # give up if the agent hangs
RATE_LIMIT = 20             # requests allowed ...
RATE_WINDOW_S = 60          # ... per this many seconds, per visitor

sessions = InMemorySessionService()
runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=sessions)
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ---------- guardrail 1: rate limit (stops one visitor running up your bill) ----------
_hits = defaultdict(deque)

# Passages retrieved earlier in each chat. If the agent answers a follow-up from memory
# without searching again, its citations can still be matched to a passage to highlight.
_session_passages = {}
MAX_PASSAGES_PER_SESSION = 40


def rate_limit(request: Request):
    visitor = request.client.host if request.client else "unknown"
    now = time.monotonic()
    recent = _hits[visitor]
    while recent and now - recent[0] > RATE_WINDOW_S:
        recent.popleft()
    if len(recent) >= RATE_LIMIT:
        raise HTTPException(429, "Too many requests. Wait a minute and try again.")
    recent.append(now)


# ---------- guardrail: daily cap for the whole app (protects your API quota) ----------
_daily = {"day": None, "count": 0}


def check_daily_limit():
    today = time.strftime("%Y-%m-%d", time.gmtime())
    if _daily["day"] != today:
        _daily.update(day=today, count=0)
    if DAILY_ASK_LIMIT and _daily["count"] >= DAILY_ASK_LIMIT:
        raise HTTPException(429, "This demo has reached its daily question limit. Please try again tomorrow.")
    _daily["count"] += 1


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
async def documents():
    names = sorted(f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith(".pdf"))
    return {"documents": names, "uploads_enabled": ALLOW_UPLOADS}


@app.post("/upload", dependencies=[Depends(rate_limit)])
async def upload(file: UploadFile):
    if not ALLOW_UPLOADS:
        raise HTTPException(403, "Uploads are turned off on this demo. Ask about the sample documents instead.")
    name = safe_name(file.filename)
    if not name.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are allowed.")

    final_path = os.path.join(UPLOAD_DIR, name)
    # Work on a temporary copy, so a failed upload can never damage a good file that is already there.
    temp_path = os.path.join(UPLOAD_DIR, f"{name}.{uuid.uuid4().hex}.part")
    try:
        if not await save_with_limit(file, temp_path):
            raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB} MB. Upload a smaller PDF.")

        # Guardrail 2: a real PDF starts with these 5 bytes, whatever the file is called.
        with open(temp_path, "rb") as f:
            if f.read(5) != b"%PDF-":
                raise HTTPException(400, "This file is not a valid PDF.")

        try:
            # Indexing is slow (it calls Gemini). Run it in a worker thread, otherwise the whole
            # server, including /health, freezes until it finishes.
            count = await asyncio.to_thread(ingest_pdf, temp_path, name)
        except ValueError as e:             # e.g. scanned PDF with no readable text
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


async def run_agent(question: str, sid: str):
    existing = await sessions.get_session(app_name=APP_NAME, user_id="user", session_id=sid)
    if not existing:
        await sessions.create_session(app_name=APP_NAME, user_id="user", session_id=sid)

    message = types.Content(role="user", parts=[types.Part(text=question)])
    answer, passages = "", []
    async for event in runner.run_async(user_id="user", session_id=sid, new_message=message):
        for fr in (event.get_function_responses() or []):
            if fr.name == "search_documents":
                passages.extend((fr.response or {}).get("passages", []))
        if event.is_final_response() and event.content and event.content.parts:
            answer = event.content.parts[0].text or ""
    return answer, passages


@app.post("/ask", dependencies=[Depends(rate_limit)])
async def ask(q: Question):
    question = q.question.strip()
    if not question:
        raise HTTPException(400, "Type a question first.")
    if len(question) > MAX_QUESTION_CHARS:
        raise HTTPException(400, f"Question is too long. Keep it under {MAX_QUESTION_CHARS} characters.")

    check_daily_limit()
    sid = q.session_id or str(uuid.uuid4())
    try:
        answer, passages = await asyncio.wait_for(run_agent(question, sid), timeout=AGENT_TIMEOUT_S)
    except asyncio.TimeoutError:
        raise HTTPException(504, "The agent took too long. Try a simpler question.")
    except Exception:
        log.exception("Agent failed")
        raise HTTPException(502, "The agent hit an error. Try again in a moment.")

    sources = merge_sources(passages, _session_passages.get(sid, []))
    _session_passages[sid] = sources

    return {
        "answer": answer or "Sorry, I couldn't produce an answer.",
        "session_id": sid,
        "sources": sources,
    }


# Order matters: specific mounts first, catch-all "/" last.
app.mount("/files", StaticFiles(directory=UPLOAD_DIR), name="files")
app.mount("/", StaticFiles(directory="static", html=True), name="static")