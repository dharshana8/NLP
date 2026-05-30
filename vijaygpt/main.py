from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

chunks = [
    "AI is transforming education",
    "RAG helps reduce hallucination",
    "Embedding convert text into vectors"
]

embedding = model.encode(chunks)

query = "What reduces hallucination?"
print(query)
query_embedding = model.encode(query)

similarities = cosine_similarity([query_embedding], embedding)

best_match_index = similarities.argmax()
print(chunks[best_match_index])
