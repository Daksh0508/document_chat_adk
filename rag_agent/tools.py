import os

import chromadb

from .ingest import gemini, EMBED_MODEL, DB_PATH, UPLOAD_DIR

MIN_SCORE = 0.45   # passages scoring lower are treated as "not found"; tune in Step 10


def _uploaded_documents():
    """The PDFs the user can actually see in the viewer."""
    if not os.path.isdir(UPLOAD_DIR):
        return []
    return [f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith(".pdf")]


def search_documents(question: str) -> dict:
    """Searches the uploaded documents and returns the most relevant passages.

    Args:
        question: A clear search query describing what to look for.

    Returns:
        A dict with status and a list of passages, each with text,
        source file, page number, and a similarity score from 0 to 1.
    """
    names = _uploaded_documents()
    if not names:
        return {"status": "error", "message": "No documents uploaded yet.", "passages": []}

    try:
        drawer = chromadb.PersistentClient(path=DB_PATH).get_collection("docs")
    except Exception:
        return {"status": "error", "message": "No documents uploaded yet.", "passages": []}

    resp = gemini.models.embed_content(model=EMBED_MODEL, contents=question)
    found = drawer.query(
        query_embeddings=[resp.embeddings[0].values],
        n_results=4,
        where={"source": {"$in": names}},    # only look inside the uploaded files
    )

    passages = [
        {"text": t, "source": m["source"], "page": m["page"], "score": round(1 - d, 3)}
        for t, m, d in zip(found["documents"][0], found["metadatas"][0], found["distances"][0])
        if (1 - d) >= MIN_SCORE
    ]
    if not passages:
        return {"status": "no_match", "message": "Nothing relevant was found in the documents.", "passages": []}
    return {"status": "success", "passages": passages}