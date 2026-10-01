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
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")


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


def ingest_pdf(path, source=None):
    """Index a PDF. `source` is the document name shown to users (defaults to the file name)."""
    name = source or os.path.basename(path)
    chunks = split_into_chunks(read_pdf(path), name)
    if not chunks:
        raise ValueError("No readable text found (scanned PDF?)")

    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts)          # do the risky part first

    db = chromadb.PersistentClient(path=DB_PATH)
    drawer = db.get_or_create_collection("docs", metadata={"hnsw:space": "cosine"})
    # Re-uploading a file replaces its old notes instead of mixing old and new.
    drawer.delete(where={"source": name})
    drawer.upsert(
        ids=[f'{name}-p{c["page"]}-{i}' for i, c in enumerate(chunks)],
        documents=texts,
        embeddings=vectors,
        metadatas=[{"source": c["source"], "page": c["page"]} for c in chunks],
    )
    return len(chunks)


if __name__ == "__main__":
    # Usage: python -m rag_agent.ingest path/to/file.pdf
    if len(sys.argv) < 2:
        print("Usage: python -m rag_agent.ingest path/to/file.pdf")
        sys.exit(1)
    print("Stored", ingest_pdf(sys.argv[1]), "chunks")