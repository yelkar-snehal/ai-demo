import openai
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_embedding(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding

def cosine_similarity(vec1, vec2):
    vec1, vec2 = np.array(vec1), np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def find_most_similar(query, documents):
    query_embedding = get_embedding(query)
    document_embeddings = [get_embedding(doc) for doc in documents]

    similarities = [cosine_similarity(query_embedding, doc_emb) for doc_emb in document_embeddings]

    best_index = np.argmax(similarities)
    return documents[best_index], similarities[best_index]

if __name__ == "__main__":
    docs = [
        "You get 18 annual leaves per year, excluding weekends.",
        "The maternity leave policy grants up to 6 months of paid leave.",
        "All employees must adhere to the work-from-home security checklist.",
        "Performance reviews happen bi-annually in June and December.",
        "We offer 12 sick leaves per calendar year."
    ]

    query = input("Ask your question: ")

    match, score = find_most_similar(query, docs)

    print("\nBest Match:\n", match)
    print("Similarity Score:", score)