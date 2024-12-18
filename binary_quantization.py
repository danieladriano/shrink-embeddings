from sentence_transformers import SentenceTransformer
from sentence_transformers.quantization import quantize_embeddings
from datasets import load_dataset


model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
dataset = load_dataset("nq_open", split="train[:1000]")["question"]

embeddings = model.encode(sentences=dataset)
binary_embeddings = quantize_embeddings(embeddings=embeddings, precision="binary")

print("Float embeddings")
print(embeddings.dtype)
print(embeddings.shape)
print(embeddings.nbytes)
print("Binary embeddings")
print(binary_embeddings.dtype)
print(binary_embeddings.shape)
print(binary_embeddings.dtype)
