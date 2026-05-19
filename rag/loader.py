from pypdf import PdfReader
from config import CHUNK_OVERLAP, CHUNK_SIZE
import os
import glob

def load_pdfs(folderpath: str) -> list[dict]:
    """Load all PDFs from a folder and return all text as a single string."""
    pdf_files = glob.glob(f"{folderpath}/*.pdf")
    
    if not pdf_files:
        raise FileNotFoundError(f"No PDF found in: {folderpath}")
    
    all_texts = []
    for filepath in pdf_files:
        reader = PdfReader(filepath)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            all_texts.append({"text": text, "source": filepath, "page": page_num+1})
    
    return all_texts

def split_text(pages: list[dict]) -> list[dict]:
    """Split text into overlapping chunks based on config parameters."""
    chunks = []
    for page in pages:
        text = page["text"]
        for i in range(0, len(text)-1, CHUNK_SIZE - CHUNK_OVERLAP):
            chunks.append({
                "text": text[i: i + CHUNK_SIZE],
                "source": page["source"],
                "page": page["page"]
            })
    return chunks