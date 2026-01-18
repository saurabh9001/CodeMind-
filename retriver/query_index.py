import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

index = faiss.read_index("vector_db/code_index.faiss")
metadata = np.load("vector_db/metadata.npy", allow_pickle=True)

model = SentenceTransformer("all-MiniLM-L6-v2")

while True:
    query = input("\nEnter your query: ")

    query_vec = model.encode([query])
    distances, indices = index.search(query_vec, k=5)

    print("\nTop matching methods:\n")

    for i in indices[0]:
        chunk = metadata[i]

        print("Method:", chunk["method"])

        if chunk.get("calls"):
            print("Calls:", [c["target"] for c in chunk["calls"]])

        if chunk.get("called_by"):
            print("Called By:", chunk["called_by"])

        print("-" * 50)
