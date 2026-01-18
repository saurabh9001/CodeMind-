import json
import os

INPUT_FILE = "COMPLETE_DEEP_ANALYSIS.json"
OUTPUT_FILE = "data/chunks.json"

os.makedirs("data", exist_ok=True)

with open(INPUT_FILE, "r") as f:
    data = json.load(f)

call_graph = data["method_call_graph"]

chunks = []
chunk_id = 0

for method, info in call_graph.items():
    chunk = {
        "id": chunk_id,
        "method": method,
        "calls": info.get("calls", []),
        "called_by": info.get("called_by", []),
        "type": "graph_chunk"
    }
    chunks.append(chunk)
    chunk_id += 1

with open(OUTPUT_FILE, "w") as f:
    json.dump(chunks, f, indent=2)

print("Chunks created:", len(chunks))
