from dotenv import load_dotenv
load_dotenv()
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, VECTOR_DB, COLLECTION_NAME
import chromadb

def get_embedder():
    """Load and return the sentence transformer embedding model."""
    return SentenceTransformer(EMBEDDING_MODEL)

embedder = get_embedder()

def embed_and_store(chunks: list[dict]) -> None:
    """
    Generate embeddings for text chunks and store them in ChromaDB.

    Args:
        chunks: List of text chunks to embed and store.
    """
    client = chromadb.PersistentClient(path=VECTOR_DB)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedder.encode(texts).tolist()
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=[{"source": chunk["source"], "page": chunk["page"]} for chunk in chunks]
    )