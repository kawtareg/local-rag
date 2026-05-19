from embedder import get_embedder
import chromadb
from config import VECTOR_DB, COLLECTION_NAME

def retrieve(query: str, n_results: int = 3) -> list[str]:
    """
    Retrieve the most relevant chunks for a given query.

    Args:
        query: The user's question.
        n_results: Number of chunks to retrieve.

    Returns:
        List of the most relevant text chunks.
    """
    client = chromadb.PersistentClient(path=VECTOR_DB)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    embedder = get_embedder()
    embedding = embedder.encode(query)
    results = collection.query(query_embeddings=[embedding.tolist()], n_results=n_results)
    return results["documents"][0]
