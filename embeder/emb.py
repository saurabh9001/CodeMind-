import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

os.makedirs("vector_db", exist_ok=True)

with open("/Users/home/Desktop/p/data/chunks.json") as f:
    chunks = json.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = []

for c in chunks:
    # create semantic text for each chunk
    text = f"""
    Method: {c['method']}
    Calls: {', '.join([call['target'] for call in c['calls']])}
    Called By: {', '.join(c['called_by'])}
    """
    texts.append(text)

embeddings = model.encode(texts)

dim = embeddings.shape[1]

index = faiss.IndexFlatL2(dim)
index.add(embeddings)

faiss.write_index(index, "vector_db/code_index.faiss")
np.save("vector_db/metadata.npy", chunks)

print("Embeddings indexed:", len(chunks))
