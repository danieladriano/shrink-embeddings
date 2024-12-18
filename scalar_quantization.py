from sentence_transformers import SentenceTransformer
from sentence_transformers.quantization import quantize_embeddings
from datasets import load_dataset

# load an embedding model
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")


corpus = load_dataset("nq_open", split="train[:1000]")["question"]
calibration_embeddings = model.encode(sentences=corpus)

embeddings = model.encode(["I am driving to the lake.", "It is a beautiful day."])
int8_embeddings = quantize_embeddings(
    embeddings=embeddings,
    precision="int8",
    calibration_embeddings=calibration_embeddings,
)

print("Float embeddings")
print(f"shape  {embeddings.shape}")
print(f"nbytes {embeddings.nbytes}")
print(f"dtype  {embeddings.dtype}")
print("Scalar embeddings")
print(f"shape  {int8_embeddings.shape}")
print(f"nbytes {int8_embeddings.nbytes}")
print(f"dtype  {int8_embeddings.dtype}")
