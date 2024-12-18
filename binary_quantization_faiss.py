import faiss
from sentence_transformers import SentenceTransformer
from sentence_transformers.quantization import quantize_embeddings

SENTENCES = [
    "The dog (Canis familiaris or Canis lupus familiaris) is a domesticated descendant of the wolf.",
    "Albert Einstein was born in Ulm in the Kingdom of Württemberg in the German Empire, on 14 March 1879.",
    "Einstein excelled at physics and mathematics from an early age, and soon acquired the mathematical expertise normally only found in a child several years his senior.",
    "Werner Karl Heisenberg was a German theoretical physicist, one of the main pioneers of the theory of quantum mechanics, and a principal scientist in the Nazi nuclear weapons program during World War II.",
    "Steven Paul Jobs (February 24, 1955 - October 5, 2011) was an American businessman, inventor, and investor best known for co-founding the technology giant Apple Inc.",
    "The cat (Felis catus), commonly referred to as the domestic cat or house cat, is the only domesticated species in the family Felidae.",
]

model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
embeddings = model.encode(sentences=SENTENCES)
binary_embeddings = quantize_embeddings(embeddings=embeddings, precision="ubinary")

num_dim = 1024
index = faiss.IndexBinaryFlat(num_dim)

index.add(binary_embeddings)

query = ["Where was Albert Einstein born?"]
query_embedding = model.encode(sentences=query)
binary_query_embedding = quantize_embeddings(
    embeddings=query_embedding.reshape(1, -1), precision="ubinary"
)

hits_scores, hits_doc_ids = index.search(binary_query_embedding, k=6)
print(f"Scores {hits_scores[0]}")
print(f"Ids    {hits_doc_ids[0]}")
print(f"Response: {SENTENCES[hits_doc_ids[0][0]]}")
