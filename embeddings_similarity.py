import openai
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()  # Load .env file
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_embedding(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def cosine_similarity(vec1, vec2):
    vec1, vec2 = np.array(vec1), np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

if __name__ == "__main__":
    text1 = "How are you doing today?"
    text2 = "What’s up?"
    text3 = "Banana and mango are fruits."

    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)
    emb3 = get_embedding(text3)

    print("Similarity (1 vs 2):", cosine_similarity(emb1, emb2))
    print("Similarity (1 vs 3):", cosine_similarity(emb1, emb3))