from pypdf import PdfReader
from config import CHUNK_OVERLAP, CHUNK_SIZE

def load_pdf(filepath: str) -> str:
    """Load a PDF file and return all text as a single string."""
    reader = PdfReader(filepath)
    all_text = []
    for page in reader.pages:
        text = page.extract_text() or ""
        all_text.append(text)
    full_txt = "\n".join(all_text)
    return full_txt

def split_text(text: str) -> list[str]:
    """Split text into overlapping chunks based on config parameters."""
    chunks = []
    for i in range (0, len(text)-1,CHUNK_SIZE-CHUNK_OVERLAP):
        chunk = text[i: i + CHUNK_SIZE]
        chunks.append(chunk)
    return chunks