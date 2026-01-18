"""
Enhanced QA System with Enriched Metadata
Uses improved chunks with comprehensive semantic descriptions
"""
import os
import faiss
import numpy as np
import openai
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


# Load environment
REPO_ROOT = Path(__file__).parent.parent
load_dotenv(dotenv_path=REPO_ROOT / ".env")


def load_index_and_metadata(index_path="vector_db_v2/code_index.faiss", metadata_path="vector_db_v2/metadata.npy"):
    """Load FAISS index and enriched metadata"""
    base_dir = REPO_ROOT
    
    index = faiss.read_index(str(base_dir / index_path))
    metadata = np.load(str(base_dir / metadata_path), allow_pickle=True)
    
    print(f"✓ Loaded FAISS index: {index.ntotal} vectors (dimension={index.d})")
    print(f"✓ Loaded metadata: {len(metadata)} enriched chunks")
    
    return index, metadata


def retrieve_top_k(index, metadata, query, k=5, use_openai_embed=True):
    """
    Retrieve top-k relevant chunks
    
    Args:
        use_openai_embed: If True, use OpenAI embeddings (matches index if created with OpenAI)
                         If False, use SentenceTransformer (for old index compatibility)
    """
    if use_openai_embed:
        # Use OpenAI embeddings (same as indexing)
        api_key = os.getenv("OPENAI_API_KEY")
        
        if api_key.startswith("sk-or-v1-"):
            base_url = "https://openrouter.ai/api/v1"
            model = "openai/text-embedding-3-large"
        else:
            base_url = None
            model = "text-embedding-3-large"
        
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        
        response = client.embeddings.create(
            input=[query],
            model=model
        )
        query_vector = np.array([response.data[0].embedding], dtype='float32')
    else:
        # Fallback: SentenceTransformer (for old index)
        model = SentenceTransformer("all-MiniLM-L6-v2")
        query_vector = model.encode([query])
    
    # Search
    distances, indices = index.search(query_vector, k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        chunk = metadata[idx]
        results.append({
            'chunk': chunk,
            'distance': float(distances[0][i]),
            'rank': i + 1
        })
    
    return results


def build_prompt(query, results):
    """Build enhanced prompt with rich metadata"""
    
    context_parts = []
    
    for r in results:
        chunk = r['chunk']
        
        # Format chunk with available metadata
        chunk_text = f"""
--- CHUNK {r['rank']}: {chunk['id']} (distance={r['distance']:.4f}) ---
Type: {chunk.get('spring_component_type', 'unknown')} ({chunk.get('component_layer', 'unknown')} layer)
File: {chunk.get('file_path', 'unknown')}
Complexity: {chunk.get('complexity', 0)} | Risk: {chunk.get('risk_level', 'unknown')}

{chunk.get('semantic_text', '')}

Metadata:
  - Calls {len(chunk.get('calls', []))} methods: {', '.join(chunk.get('calls', [])[:10])}
  - Called by {len(chunk.get('called_by', []))} methods: {', '.join(chunk.get('called_by', [])[:5])}
  - Dependencies: {', '.join(chunk.get('class_dependencies', [])[:5])}
  - DB Operations: {len(chunk.get('database_operations', []))}
  - Scheduled: {chunk.get('is_scheduled', False)}
  - Error Handling: {len(chunk.get('error_handling', []))} patterns
"""
        
        # Add DB operation details if present
        if chunk.get('database_operations'):
            chunk_text += "\n  Database Operations:\n"
            for op in chunk['database_operations'][:3]:
                chunk_text += f"    - {op}\n"
        
        # Add scheduled task info
        if chunk.get('is_scheduled'):
            chunk_text += f"\n  Scheduled Info: {chunk.get('scheduled_info', {})}\n"
        
        context_parts.append(chunk_text)
    
    context = "\n".join(context_parts)
    
    prompt = f"""You are a code analysis assistant. Answer the user's question about this Java codebase.

Context (Top {len(results)} most relevant code methods):
{context}

User Question: {query}

Instructions:
- Provide a clear, technical answer based on the retrieved code context
- Reference specific methods, classes, and relationships
- If describing logic, explain the call flow and dependencies
- If the context doesn't fully answer the question, state what's available and what's missing
- Use the metadata (complexity, DB ops, error handling) to provide deeper insights

Answer:"""
    
    return prompt


def call_openai(prompt):
    """Call OpenAI API with OpenRouter detection"""
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        return "Error: OPENAI_API_KEY not found in environment"
    
    # Detect endpoint
    if api_key.startswith("sk-or-v1-"):
        base_url = "https://openrouter.ai/api/v1"
        model = "openai/gpt-3.5-turbo"
    else:
        base_url = None
        model = "gpt-3.5-turbo"
    
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful code analysis assistant specialized in Java Spring applications."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.3
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return f"Error calling LLM: {str(e)}"


def interactive_loop():
    """Interactive QA loop"""
    print("\n" + "="*70)
    print("Enhanced Code Intelligence QA System")
    print("Using enriched metadata with comprehensive semantic understanding")
    print("="*70)
    
    # Load index
    index, metadata = load_index_and_metadata()
    
    print("\nReady! Ask questions about the codebase.")
    print("Examples:")
    print("  - How does failure handling work?")
    print("  - What methods interact with the database?")
    print("  - Show me scheduled tasks")
    print("  - What are the high-complexity methods?")
    print("  - Explain the ExecutionMode class")
    print("\nType 'quit' to exit.\n")
    
    while True:
        query = input("Your question: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not query:
            continue
        
        print("\nSearching code intelligence index...")
        
        # Retrieve relevant chunks
        results = retrieve_top_k(index, metadata, query, k=5, use_openai_embed=True)
        
        # Show brief retrieval results
        print(f"\nTop {len(results)} relevant code units:")
        for r in results:
            chunk = r['chunk']
            print(f"  {r['rank']}. {chunk['id']} (distance={r['distance']:.4f})")
            print(f"     {chunk.get('spring_component_type', 'unknown')} | complexity={chunk.get('complexity', 0)} | {len(chunk.get('calls', []))} calls")
        
        # Build prompt
        prompt = build_prompt(query, results)
        
        # Call LLM
        print("\nAnalyzing with LLM...\n")
        answer = call_openai(prompt)
        
        print("="*70)
        print("ANSWER:")
        print("="*70)
        print(answer)
        print("="*70 + "\n")


if __name__ == "__main__":
    interactive_loop()
