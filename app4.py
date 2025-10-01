from langchain_ollama import OllamaEmbeddings
import numpy as np

# Initialize embeddings model with specific parameters
embeddings = OllamaEmbeddings(
    model="nomic-embed-text:v1.5",  # Specialized embedding model that is also supported by Ollama
    base_url="http://localhost:11434",
)

# Create embeddings for a query
query = "How do neural networks learn?"
query_embedding = embeddings.embed_query(query)
print(f"Embedding dimension: {len(query_embedding)}")

# Create embeddings for multiple documents
documents = [
    "Neural networks learn through backpropagation",
    "Transformers use attention mechanisms",
    "LLMs are trained on text data"
]

doc_embeddings = embeddings.embed_documents(documents)

# Calculate similarity between vectors
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Find most similar document to query
similarities = [cosine_similarity(query_embedding, doc_emb) for doc_emb in doc_embeddings]
most_similar_idx = np.argmax(similarities)
print(f"Most similar document: {documents[most_similar_idx]}")
print(f"Similarity score: {similarities[most_similar_idx]:.3f}")


