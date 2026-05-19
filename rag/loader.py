from pypdf import PdfReader
from config import CHUNK_OVERLAP, CHUNK_SIZE
import os
import glob

def load_pdfs(folderpath: str) -> str:
    """Load all PDFs from a folder and return all text as a single string."""
    pdf_files = glob.glob(f"{folderpath}/*.pdf")
    
    if not pdf_files:
        raise FileNotFoundError(f"No PDF found in: {folderpath}")
    
    all_texts = []
    for filepath in pdf_files:
        reader = PdfReader(filepath)
        for page in reader.pages:
            text = page.extract_text() or ""
            all_texts.append(text)
    
    return "\n".join(all_texts)

def split_text(text: str) -> list[str]:
    """Split text into overlapping chunks based on config parameters."""
    chunks = []
    for i in range (0, len(text)-1,CHUNK_SIZE-CHUNK_OVERLAP):
        chunk = text[i: i + CHUNK_SIZE]
        chunks.append(chunk)
    return chunks