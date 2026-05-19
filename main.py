import argparse 
from rag.loader import load_pdf, split_text
from rag.embedder import embed_and_store
from rag.retriever import retrieve
from rag.generator import generate
from openai import APIConnectionError

def entire_pipeline():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=str, required=True)
    args = parser.parse_args()

    try:
        print(f"Loading {args.pdf}...")
        full_pdf = load_pdf(args.pdf)
        chunks = split_text(full_pdf)
        print(f"{len(chunks)} chunks created")
        embed_and_store(chunks)
        print("Ready! Type 'quit' to exit\n")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    while True:
        query = input(">>>: ")
        if not query.strip():
            continue
        if query == "quit":
            break
        context_chunks = retrieve(query)
        
        try:
            reply = generate(query, context_chunks=context_chunks)
            print(f"\n{reply}\n")
        except APIConnectionError:
            print("Erreur : Ollama n'est pas lancé. Fais 'brew services start ollama'")
        except Exception as e:
            print("Erreur innatendue: ", e)

if __name__ == "__main__":
    entire_pipeline()