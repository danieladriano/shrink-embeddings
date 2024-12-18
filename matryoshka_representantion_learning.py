import faiss
import torch
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MATRYOSHKA_DIM = 64
WIKIPEDIA_TEXTS = [
    "The dog (Canis familiaris or Canis lupus familiaris) is a domesticated descendant of the wolf.",
    "Albert Einstein was born in Ulm in the Kingdom of Württemberg in the German Empire, on 14 March 1879.",
    "Einstein excelled at physics and mathematics from an early age, and soon acquired the mathematical expertise normally only found in a child several years his senior.",
    "Werner Karl Heisenberg was a German theoretical physicist, one of the main pioneers of the theory of quantum mechanics, and a principal scientist in the Nazi nuclear weapons program during World War II.",
    "Steven Paul Jobs (February 24, 1955 - October 5, 2011) was an American businessman, inventor, and investor best known for co-founding the technology giant Apple Inc.",
    "The cat (Felis catus), commonly referred to as the domestic cat or house cat, is the only domesticated species in the family Felidae.",
]

device = "cuda" if torch.cuda.is_available() else "mps"
logger.info(f"Executing on device {device}")
model = SentenceTransformer(
    model_name_or_path="nomic-ai/nomic-embed-text-v1.5",
    device=device,
    trust_remote_code=True,
    prompts={
        "search_query": "search_query: ",
        "search_document": "search_document: ",
        "classification": "classification: ",
        "clustering": "clustering: ",
    },
)
logger.info("Model defined")

index = faiss.IndexFlatIP(MATRYOSHKA_DIM)
logger.info("Faiss defined")


def embed_sentences(
    model: SentenceTransformer,
    setences: list[str],
    prompt_name: str,
    matryoshka_dim: int,
    device: str,
) -> torch.Tensor:
    if matryoshka_dim > 768:
        raise Exception("Maximim dimension for nomic-embed-text-v1.5 is 768")

    embeddings = model.encode(
        sentences=setences,
        prompt_name=prompt_name,
        device=device,
        convert_to_tensor=True,
    )

    embeddings = torch.nn.functional.layer_norm(
        input=embeddings, normalized_shape=(embeddings.shape[1],)
    )

    embeddings = embeddings[:, :matryoshka_dim]
    embeddings = torch.nn.functional.normalize(input=embeddings, p=2, dim=1)
    return embeddings.cpu()


def set_embeddings() -> None:
    logger.info("Setting embeddings")
    document_embeddings = embed_sentences(
        model=model,
        setences=WIKIPEDIA_TEXTS,
        prompt_name="search_document",
        matryoshka_dim=MATRYOSHKA_DIM,
        device=device,
    )

    index.add(document_embeddings)


def main():
    set_embeddings()

    question = ["Where was Albert Einstein born?"]
    logger.info(f"Question: {question[0]}")

    question_embedding = embed_sentences(
        model=model,
        setences=question,
        prompt_name="search_query",
        matryoshka_dim=MATRYOSHKA_DIM,
        device=device,
    )

    distances, indices = index.search(question_embedding, k=6)
    logger.info(f"Indices: {indices}")
    logger.info(f"Distances: {distances}")
    logger.info(f"Response: {WIKIPEDIA_TEXTS[indices[0][0]]}")


if __name__ == "__main__":
    main()
