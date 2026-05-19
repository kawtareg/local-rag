import argparse 
import glob
from rag.loader import load_pdfs, split_text
from rag.embedder import embed_and_store
from rag.retriever import retrieve
from rag.generator import generate
from openai import APIConnectionError

def entire_pipeline():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    args = parser.parse_args()

    try:
        print(f"Chargement de {args.folder}...")
        full_pdf = load_pdfs(args.folder)
        chunks = split_text(full_pdf)
        print(f"{len(chunks)} chunks crées")
        embed_and_store(chunks)
        print("Prêt! 'quit' pour quitter\n")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    history = []
    while True:
        query = input(">>>: ")
        if not query.strip():
            continue
        if query == "quit":
            break
        context_chunks = retrieve(query)

        try:
            reply = generate(query, context_chunks=context_chunks, history=history)
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": reply})
            print(f"\n{reply}\n")
            print("Sources :")
            for i, chunk in enumerate(context_chunks):
                clean = chunk.replace("\n", " ").strip()
                print(f"  [{i+1}] \"{clean[:100]}...\"")
        except APIConnectionError:
            print("Erreur : Ollama n'est pas lancé. Fais 'brew services start ollama'")
        except Exception as e:
            print("Erreur innatendue: ", e)

if __name__ == "__main__":
    entire_pipeline()