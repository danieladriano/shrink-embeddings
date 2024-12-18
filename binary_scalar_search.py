import faiss
from sentence_transformers import SentenceTransformer
from sentence_transformers.quantization import quantize_embeddings
from datasets import load_dataset
from usearch.index import Index
import time
import logging

BINARY_INDEX_PATH = "./db/wikipedia_ubinary_faiss_50m.index"
SCALAR_INDEX_PATH = "./db/wikipedia_int8_usearch_50m.index"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BinaryScalarSearch:
    def __init__(self):
        logger.info("Loading wikipedia dataset")
        self.wikipedia_dataset = load_dataset(
            "mixedbread-ai/wikipedia-data-en-2023-11", split="train", num_proc=4
        ).select_columns(["title", "text"])

        logger.info("Loading ubinary faiss index")
        self.binary_index: faiss.IndexBinaryFlat = faiss.read_index_binary(
            BINARY_INDEX_PATH
        )

        logger.info("Loading scalar usearch index")
        self.scalar_index_view: Index = Index.restore(SCALAR_INDEX_PATH, view=True)

        logger.info("Loading embedding model")
        self.model = SentenceTransformer(
            model_name_or_path="mixedbread-ai/mxbai-embed-large-v1",
            prompts={
                "retrieval": "Represent this sentence for searching relevant passages: "
            },
            default_prompt_name="retrieval",
        )

    def search(self, query: str, top_k: int = 100) -> list[str]:
        query_embedding = self.model.encode(sentences=query)

        query_embedding_ubinary = quantize_embeddings(
            embeddings=query_embedding.reshape(1, -1), precision="ubinary"
        )

        start_time = time.time()
        bscores, binary_ids = self.binary_index.search(query_embedding_ubinary, top_k)
        logger.info(f"Binary search time {time.time() - start_time}")
        logger.info(f"Binary scores {bscores[0][:5]} - ids {binary_ids[0][:5]}")
        binary_ids = binary_ids[0]

        start_time = time.time()
        scalar_embedding = self.scalar_index_view[binary_ids].astype(int)
        logger.info(f"Scalar (int8) search time {time.time() - start_time}")

        start_time = time.time()
        scores = query_embedding @ scalar_embedding.T
        logger.info(f"Rescore time {time.time() - start_time}")

        indices = scores.argsort()[::-1][:top_k]
        top_k_indices = binary_ids[indices]

        logger.info(f"Final scores {scores[:5]} - ids {top_k_indices[:5]}")

        top_k_text = [
            self.wikipedia_dataset[idx]["text"] for idx in top_k_indices.tolist()
        ]
        return top_k_text


def main():
    binary_scalar_search = BinaryScalarSearch()

    while True:
        query = input(f"Insert query: ")
        if query == "quit":
            break
        results = binary_scalar_search.search(query=query)
        print(results[0])


if __name__ == "__main__":
    main()
