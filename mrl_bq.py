from faiss import IndexBinaryFlat, write_index_binary, IndexHNSWSQ
import faiss
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.quantization import quantize_embeddings
import logging
from datasets import load_dataset
import numpy as np
from usearch.index import Index

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MATRYOSHKA_DIM = 64


class MatryoshkaRepresentantionAndBinaryQuantization:
    def __init__(self):
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
        self.dataset = load_dataset(
            "mixedbread-ai/wikipedia-data-en-2023-11", split="train[:10000]", num_proc=4
        )["text"]
        self.binary_index = IndexBinaryFlat(MATRYOSHKA_DIM)
        self.scalar_index = Index(ndim=MATRYOSHKA_DIM, metric="ip", dtype="i8")

    def get_matryoshka_embeddings(self) -> torch.Tensor:
        embeddings = self.model.encode(
            sentences=self.dataset,
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

    def get_binary_embeddings(self, embeddings: torch.Tensor) -> np.ndarray:
        return quantize_embeddings(embeddings=embeddings, precision="ubinary")

    def get_scalar_embeddings(self, embeddings: torch.Tensor) -> np.ndarray:
        return quantize_embeddings(embeddings=embeddings, precision="int8")

    def create_embeddings(self) -> None:
        embeddings = self.get_matryoshka_embeddings()
        binary_embeddings = self.get_binary_embeddings(embeddings=embeddings)
        scalar_embeddings = self.get_scalar_embeddings(embeddings=embeddings)
        logger.info(
            f"Matryoshka {embeddings.shape} - {embeddings.dtype} - {embeddings.nbytes}"
        )

        logger.info(
            f"Binary {binary_embeddings.shape} - {binary_embeddings.dtype} - {binary_embeddings.nbytes}"
        )
        self.binary_index.add(binary_embeddings)
        write_index_binary(
            self.binary_index, "./db/wikipedia_ubinary_faiss_64_10k.index"
        )
        logger.info(
            f"Scalar {scalar_embeddings.shape} - {scalar_embeddings.dtype} - {scalar_embeddings.nbytes}"
        )
        self.scalar_index.add(np.arange(len(scalar_embeddings)), scalar_embeddings)
        self.scalar_index.save("./db/wikipedia_int8_usearh_64_10k.index")


def main():
    mrl_bq_embeddings = MatryoshkaRepresentantionAndBinaryQuantization()
    mrl_bq_embeddings.create_embeddings()


if __name__ == "__main__":
    main()
