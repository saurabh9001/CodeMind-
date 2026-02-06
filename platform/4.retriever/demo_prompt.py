"""
Demo: Show Complete LLM Prompt Format
This script demonstrates what the complete prompt looks like
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from retriever_raw import load_index_and_metadata, retrieve_top_k, build_llm_prompt

# Load index
print("Loading index...")
index, metadata = load_index_and_metadata()

# Example query
query = "what are scheduled tasks"
print(f"\nQuery: {query}\n")

# Retrieve chunks
print("Retrieving relevant chunks...")
results = retrieve_top_k(index, metadata, query, k=5, use_openai_embed=True)

# Build and display complete prompt
print("\n" + "="*80)
print("COMPLETE LLM PROMPT (exactly what gets sent to the API)")
print("="*80 + "\n")

complete_prompt = build_llm_prompt(query, results)
print(complete_prompt)

print("\n" + "="*80)
print("This is the EXACT text that would be sent to ChatGPT/Claude/etc.")
print("="*80)
