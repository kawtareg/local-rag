from dotenv import load_dotenv
load_dotenv()
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from rag.embedder import get_embedder
import chromadb
from config import VECTOR_DB, COLLECTION_NAME

embedder = get_embedder()

def retrieve(query: str, n_results: int = 3) -> list[dict]:
    """
    Retrieve the most relevant chunks for a given query.

    Args:
        query: The user's question.
        n_results: Number of chunks to retrieve.

    Returns:
        List of dicts with keys 'text', 'source', and 'page'.
    """
    client = chromadb.PersistentClient(path=VECTOR_DB)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    embedding = embedder.encode(query)
    results = collection.query(query_embeddings=[embedding.tolist()], n_results=n_results)
    return [
    {
        "text": doc,
        "source": meta["source"],
        "page": meta["page"]
    }
    for doc, meta in zip(results["documents"][0], results["metadatas"][0])
]