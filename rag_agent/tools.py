import chromadb
from google.adk.tools import ToolContext

from .ingest import gemini, EMBED_MODEL, DB_PATH, SHARED_OWNER, visible_documents

MIN_SCORE = 0.45   # passages scoring lower are treated as "not found"; tune with your evaluation


def search_documents(question: str, tool_context: ToolContext = None) -> dict:
    """Searches the documents the user can see and returns the most relevant passages.

    Args:
        question: A clear search query describing what to look for.

    Returns:
        A dict with status and a list of passages, each with text,
        source file, page number, and a similarity score from 0 to 1.
    """
    # ADK fills in tool_context. Its user_id is the visitor, so each visitor only searches
    # their own uploads plus the shared sample documents.
    visitor = getattr(tool_context, "user_id", None)
    names = visible_documents(visitor)
    if not names:
        return {"status": "error", "message": "No documents uploaded yet.", "passages": []}

    try:
        drawer = chromadb.PersistentClient(path=DB_PATH).get_collection("docs")
    except Exception:
        return {"status": "error", "message": "No documents uploaded yet.", "passages": []}

    owners = [SHARED_OWNER] + ([visitor] if visitor else [])
    resp = gemini.models.embed_content(model=EMBED_MODEL, contents=question)
    found = drawer.query(
        query_embeddings=[resp.embeddings[0].values],
        n_results=4,
        where={"$and": [{"owner": {"$in": owners}}, {"source": {"$in": names}}]},
    )

    passages = [
        {"text": t, "source": m["source"], "page": m["page"], "score": round(1 - d, 3)}
        for t, m, d in zip(found["documents"][0], found["metadatas"][0], found["distances"][0])
        if (1 - d) >= MIN_SCORE
    ]
    if not passages:
        return {"status": "no_match", "message": "Nothing relevant was found in the documents.", "passages": []}
    return {"status": "success", "passages": passages}
