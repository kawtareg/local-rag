from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from rag.retriever import retrieve
from rag.generator import generate
from config import EVALUATING_MODEL

test_dataset = [
    {
        "question": "Combien de couches compose l'architecture TCP/IP ?",
        "ground_truth": "L'architecture TCP/IP est structurée en 4 couches : la couche lien, la couche réseau, la couche transport et la couche application."
    },
    {
        "question": "Quelle est la différence entre une adresse classful et classless ?",
        "ground_truth": "En classful, la classe de l'adresse (A, B, C) détermine le nombre de bits réservés à l'identifiant réseau. En classless (CIDR), on utilise un suffixe explicite (notation IP/nb de bits) pour identifier les bits réseau, permettant des tailles de réseaux mieux adaptées et évitant le gaspillage d'adresses."
    },
    {
        "question": "Qu'est-ce que le DNS et à quoi sert-il ?",
        "ground_truth": "Le DNS (Domain Name Server) est un système de bases de données distribuées qui effectue la conversion entre les noms de machines et les adresses IP. C'est un espace de noms hiérarchisé représenté sous forme d'arbre."
    },
    {
        "question": "Quand a été créé ARPANET et quels étaient ses premiers noeuds ?",
        "ground_truth": "ARPANET a été créé en 1969 avec l'interconnexion de quatre noeuds : l'Université de Californie UCLA, l'Institut de Recherche à Stanford SRI, l'Université de Santa Barbara UCSB et l'Université de l'Utah."
    },
    {
        "question": "Qu'est-ce qu'un masque de sous-réseau ?",
        "ground_truth": "Un masque de sous-réseau est un mot de 32 bits ajouté en interne à une adresse IPv4 permettant de découper un réseau en plusieurs sous-réseaux. Les bits sont à 1 pour l'identifiant réseau et à 0 pour l'identifiant machine. Il n'a de sens qu'au sein du réseau, pas sur Internet."
    },
]

def evaluate_rag(test_dataset: list[dict]) -> None:
    """Evaluate the RAG pipeline using RAGAS metrics.

    Retrieves context chunks and generates answers for each question in the
    test dataset, then evaluates the results using RAGAS metrics including
    faithfulness, answer relevancy, context precision and context recall.

    Args:
        test_dataset: List of dicts with keys 'question' and 'ground_truth'.
    """
    results = []
    for item in test_dataset:
        question = item["question"]
        ground_truth = item["ground_truth"]
        chunks = retrieve(question, n_results=5)
        contexts = [chunk["text"] for chunk in chunks]
        answer = generate(question, contexts, history=[])
        print(f"Q: {question[:50]}... ✓")
        results.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth
        })

    llm = LangchainLLMWrapper(Ollama(model=EVALUATING_MODEL))
    embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=EVALUATING_MODEL))

    dataset = Dataset.from_list(results)
    scores = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ],
        llm=llm,
        embeddings=embeddings
    )
    print("\n=== RAGAS Scores ===")
    print(scores)

if __name__ == "__main__":
    evaluate_rag(test_dataset)