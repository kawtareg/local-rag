from openai import OpenAI
from dotenv import load_dotenv
from config import MODEL, TEMPERATURE
import os
import httpx

os.environ["NO_PROXY"] = "*"

load_dotenv()

client = OpenAI(
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama",
    http_client=httpx.Client(proxy=None),
)

def generate(query: str, context_chunks: list[str], history: list[str]) -> str:
    """
    Generate an answer based on retrieved context chunks.

    Args:
        query: The user's question.
        context_chunks: List of relevant text chunks from the retriever.
        history: The past generated answers.

    Returns:
        The generated answer as a string.
    """
    context = "\n\n".join(context_chunks)
    prompt = f"""Tu es un assistant qui répond uniquement en te basant sur le contexte fourni.
        Si la réponse n'est pas dans le contexte, dis-le explicitement.
        Contexte :
        {context}
        Question : {query}
        """
    response = client.chat.completions.create(
            model=MODEL,
            temperature= TEMPERATURE,
            messages = [
                {"role": "system", "content": prompt},
                *history,
                {"role": "user", "content": query}
            ])
    return response.choices[0].message.content