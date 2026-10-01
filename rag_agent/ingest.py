import os
import sys

import chromadb
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader

load_dotenv()
gemini = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
EMBED_MODEL = "gemini-embedding-001"   # check Google's docs for the current name

# Absolute paths, so it never matters which folder you start the server from.
# Set DOCCHAT_DATA_DIR to keep data elsewhere (tests use a temp folder; a host can use a persistent disk).
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.abspath(os.getenv("DOCCHAT_DATA_DIR", BASE_DIR))
DB_PATH = os.path.join(DATA_DIR, "chroma_db")
SHARED_DIR = os.path.join(DATA_DIR, "shared")   # sample documents that every visitor can see
USERS_DIR = os.path.join(DATA_DIR, "users")     # one private folder per visitor
SHARED_OWNER = "demo"                           # the "owner" tag on shared documents

MAX_CHUNKS_PER_DOC = int(os.getenv("MAX_CHUNKS_PER_DOC", "400"))   # protects your embedding quota


# ---------- where documents live, and who may see them ----------
def user_dir(owner):
    return SHARED_DIR if owner == SHARED_OWNER else os.path.join(USERS_DIR, owner)


def pdfs_in(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(".pdf"))


def visible_documents(visitor):
    """Shared sample documents plus this visitor's own. Nobody else's."""
    names = set(pdfs_in(SHARED_DIR))
    if visitor:
        names |= set(pdfs_in(user_dir(visitor)))
    return sorted(names)


# ---------- reading and indexing ----------
def read_pdf(path):
    reader = PdfReader(path)
    return [{"page": n, "text": p.extract_text() or ""}
            for n, p in enumerate(reader.pages, start=1)]


def split_into_chunks(pages, source, size=800, overlap=100):
    chunks = []
    for page in pages:
        text, start = page["text"], 0
        while start < len(text):
            piece = text[start:start + size]
            if piece.strip():
                chunks.append({"text": piece, "source": source, "page": page["page"]})
            start += size - overlap
    return chunks


def embed_texts(texts):
    vectors = []
    for i in range(0, len(texts), 50):
        resp = gemini.models.embed_content(model=EMBED_MODEL, contents=texts[i:i + 50])
        vectors += [e.values for e in resp.embeddings]
    return vectors


def _collection():
    db = chromadb.PersistentClient(path=DB_PATH)
    return db.get_or_create_collection("docs", metadata={"hnsw:space": "cosine"})


def ingest_pdf(path, source=None, owner=SHARED_OWNER):
    """Index a PDF for one owner. `source` is the name shown to users (defaults to the file name)."""
    name = source or os.path.basename(path)
    chunks = split_into_chunks(read_pdf(path), name)
    if not chunks:
        raise ValueError("No readable text found (scanned PDF?)")
    if len(chunks) > MAX_CHUNKS_PER_DOC:
        raise ValueError("This PDF is too long to index here. Try a shorter document or a few chapters.")

    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts)          # do the risky part first

    drawer = _collection()
    # Re-uploading a file replaces its old notes (this owner's only), instead of mixing old and new.
    drawer.delete(where={"$and": [{"source": name}, {"owner": owner}]})
    drawer.upsert(
        ids=[f'{owner}/{name}-p{c["page"]}-{i}' for i, c in enumerate(chunks)],
        documents=texts,
        embeddings=vectors,
        metadatas=[{"source": c["source"], "page": c["page"], "owner": owner} for c in chunks],
    )
    return len(chunks)


def delete_document_chunks(source, owner):
    _collection().delete(where={"$and": [{"source": source}, {"owner": owner}]})


def delete_owner_chunks(owner):
    _collection().delete(where={"owner": owner})


if __name__ == "__main__":
    # Usage: python -m rag_agent.ingest path/to/file.pdf   (indexes it as a shared sample document)
    if len(sys.argv) < 2:
        print("Usage: python -m rag_agent.ingest path/to/file.pdf")
        sys.exit(1)
    print("Stored", ingest_pdf(sys.argv[1]), "chunks")
