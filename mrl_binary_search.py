import faiss
from sentence_transformers import SentenceTransformer
from sentence_transformers.quantization import quantize_embeddings
from datasets import load_dataset
from usearch.index import Index
import time
import logging
import torch

MATRYOSHKA_DIM = 64
BINARY_INDEX_PATH = f"./db/mrl_bq/wikipedia_ubinary_faiss_{MATRYOSHKA_DIM}_10k.index"


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

        logger.info("Loading embedding model")
        self.device = "cuda" if torch.cuda.is_available() else "mps"
        self.model = SentenceTransformer(
            model_name_or_path="mixedbread-ai/mxbai-embed-large-v1",
            device=self.device,
            trust_remote_code=True,
            prompts={
                "search_query": "search_query: ",
                "search_document": "search_document: ",
                "classification": "classification: ",
                "clustering": "clustering: ",
            },
        )

    def get_matryoshka_embeddings(self, query: list[str]) -> torch.Tensor:
        embeddings = self.model.encode(
            sentences=query,
            prompt_name="search_document",
            device=self.device,
            convert_to_tensor=True,
        )

        embeddings = torch.nn.functional.layer_norm(
            input=embeddings, normalized_shape=(embeddings.shape[1],)
        )

        embeddings = embeddings[:, :MATRYOSHKA_DIM]
        embeddings = torch.nn.functional.normalize(input=embeddings, p=2, dim=1)
        return embeddings.cpu()

    def search(self, query: str, top_k: int = 100) -> list[str]:
        query_embedding = self.get_matryoshka_embeddings(query=[query])
        query_embedding_ubinary = quantize_embeddings(
            embeddings=query_embedding, precision="ubinary"
        )

        start_time = time.time()
        bscores, binary_ids = self.binary_index.search(query_embedding_ubinary, top_k)
        logger.info(f"Binary search time {time.time() - start_time}")
        logger.info(f"Binary scores {bscores[0][:5]} - ids {binary_ids[0][:5]}")
        binary_ids = binary_ids[0]

        top_k_text = [
            self.wikipedia_dataset[idx]["text"] for idx in binary_ids.tolist()
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
