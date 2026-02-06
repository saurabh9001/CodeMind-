"""
Code-Aware Embedder with SOURCE CODE Support
Understands code semantics, data flow, relationships, and actual implementation
"""
import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm


def create_codebert_embeddings(chunks_path: str, output_dir: str, use_openai: bool = False):
    """
    Create code-aware embeddings using GraphCodeBERT or OpenAI
    Now optimized for chunks with source code
    
    Args:
        chunks_path: Path to enriched_chunks.json (with source code)
        output_dir: Directory to save FAISS index and metadata
        use_openai: If True, use OpenAI embeddings API (simpler, no local model)
    """
    
    print(f"Loading chunks from {chunks_path}...")
    with open(chunks_path, 'r') as f:
        chunks = json.load(f)
    
    print(f"Loaded {len(chunks)} chunks")
    
    # Analyze chunks
    chunks_with_source = sum(1 for c in chunks if c.get('source_code'))
    print(f"  ✓ {chunks_with_source} chunks have source code")
    print(f"  ✓ {len(chunks) - chunks_with_source} chunks have metadata only")
    
    if use_openai:
        embeddings = _embed_with_openai(chunks)
    else:
        embeddings = _embed_with_graphcodebert(chunks)
    
    # Create FAISS index
    print("\nBuilding FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    faiss.write_index(index, str(output_path / "code_index.faiss"))
    
    # Save metadata with source code
    with open(output_path / "metadata.json", 'w') as f:
        json.dump(chunks, f, indent=2)
    
    # Also save as numpy for backward compatibility
    np.save(str(output_path / "metadata.npy"), np.array(chunks, dtype=object))
    
    # Save embedding stats
    stats = {
        'total_chunks': len(chunks),
        'chunks_with_source': chunks_with_source,
        'embedding_dimension': dimension,
        'model_used': 'OpenAI' if use_openai else 'GraphCodeBERT',
        'avg_source_lines': sum(c.get('lines_of_code', 0) for c in chunks) / len(chunks),
        'component_breakdown': _get_component_stats(chunks)
    }
    
    with open(output_path / "embedding_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✅ Created FAISS index with {len(chunks)} vectors (dimension={dimension})")
    print(f"✓ Saved to {output_dir}/")
    print(f"\nEmbedding Statistics:")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  With source code: {stats['chunks_with_source']}")
    print(f"  Avg LOC per method: {stats['avg_source_lines']:.1f}")
    print(f"  Embedding dimension: {stats['embedding_dimension']}")
    
    return index, chunks


def _prepare_embedding_text(chunk: Dict) -> str:
    """
    Prepare optimized text for embedding that includes source code
    Balances semantic metadata with actual code implementation
    """
    parts = []
    
    # 1. Method identification
    parts.append(f"Method: {chunk['id']}")
    parts.append(f"Class: {chunk['class_name']} ({chunk.get('spring_component_type', 'Component')})")
    
    # 2. Package and layer context
    if chunk.get('class_package'):
        parts.append(f"Package: {chunk['class_package']}")
    parts.append(f"Layer: {chunk.get('component_layer', 'unknown')}")
    
    # 3. Annotations (important for understanding behavior)
    if chunk.get('source_annotations'):
        parts.append(f"Annotations: {', '.join(chunk['source_annotations'])}")
    
    # 4. Method signature (provides interface understanding)
    if chunk.get('source_signature'):
        parts.append(f"\nSignature:\n{chunk['source_signature']}")
    
    # 5. SOURCE CODE (the key addition!)
    # Include full source code for better semantic understanding
    if chunk.get('source_code'):
        source = chunk['source_code']
        
        # For very long methods, include full code but prioritize first part
        if len(source) > 5000:
            # Take first 4000 chars + last 1000 chars
            parts.append(f"\nSource Code:\n{source[:4000]}\n...\n{source[-1000:]}")
        else:
            parts.append(f"\nSource Code:\n{source}")
    
    # 6. Behavioral metadata
    behavior_tags = []
    
    if chunk.get('database_operations'):
        ops = chunk['database_operations']
        op_types = set(op.get('type', 'DB') for op in ops[:5])
        behavior_tags.append(f"Database: {', '.join(op_types)}")
    
    if chunk.get('external_api_calls'):
        apis = chunk['external_api_calls']
        api_types = set(api.get('client_type', 'HTTP') for api in apis[:3])
        behavior_tags.append(f"External APIs: {', '.join(api_types)}")
    
    if chunk.get('is_scheduled'):
        schedule_info = chunk.get('scheduled_info', {})
        if 'config' in schedule_info and 'cron' in schedule_info['config']:
            behavior_tags.append(f"Scheduled: {schedule_info['config']['cron']}")
        else:
            behavior_tags.append(f"Scheduled: {schedule_info.get('type', 'task')}")
    
    if chunk.get('error_handling'):
        exceptions = set(e.get('exception_type', 'Exception') for e in chunk['error_handling'][:3])
        behavior_tags.append(f"Handles: {', '.join(exceptions)}")
    
    if behavior_tags:
        parts.append(f"\nBehavior: {' | '.join(behavior_tags)}")
    
    # 7. Call relationships (understanding dependencies)
    calls = chunk.get('calls', [])
    if calls:
        # Include up to 10 most important calls
        call_targets = calls[:10]
        parts.append(f"\nCalls: {', '.join(call_targets)}")
    
    called_by = chunk.get('called_by', [])
    if called_by:
        parts.append(f"Called by: {', '.join(called_by[:10])}")
    
    # 8. Dependencies
    deps = chunk.get('class_dependencies', [])
    if deps:
        parts.append(f"Dependencies: {', '.join(deps[:8])}")
    
    # 9. Search keywords (for better retrieval)
    keywords = chunk.get('search_keywords', [])
    if keywords:
        parts.append(f"\nKeywords: {', '.join(keywords[:20])}")
    
    return '\n'.join(parts)


def _embed_with_graphcodebert(chunks: List[Dict]) -> np.ndarray:
    """
    Embed using GraphCodeBERT (Microsoft's code-aware model)
    Optimized for source code understanding
    """
    
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
    except ImportError:
        print("\n⚠️  transformers not installed. Installing now...")
        import subprocess
        subprocess.run(["pip", "install", "transformers", "torch"], check=True)
        from transformers import AutoTokenizer, AutoModel
        import torch
    
    print("\nLoading GraphCodeBERT model...")
    model_name = "microsoft/graphcodebert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    
    print("Generating code-aware embeddings...")
    embeddings = []
    
    batch_size = 8  # Smaller batch size due to longer text with source code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    print(f"Using device: {device}")
    
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding with GraphCodeBERT"):
        batch = chunks[i:i+batch_size]
        
        # Prepare texts with source code
        texts = [_prepare_embedding_text(chunk) for chunk in batch]
        
        # Tokenize with truncation (GraphCodeBERT max: 512 tokens)
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            # Mean pooling over sequence length
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        embeddings.append(batch_embeddings)
    
    return np.vstack(embeddings)


def _embed_with_openai(chunks: List[Dict]) -> np.ndarray:
    """
    Embed using OpenAI embeddings API
    Supports much longer context (8K tokens) - better for source code
    """
    
    try:
        import openai
        from dotenv import load_dotenv
        import os
    except ImportError:
        print("\n⚠️  openai or python-dotenv not installed")
        print("Installing required packages...")
        import subprocess
        subprocess.run(["pip", "install", "openai", "python-dotenv"], check=True)
        import openai
        from dotenv import load_dotenv
        import os
    
    # Load API key
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in .env file\n"
            "Please create a .env file with: OPENAI_API_KEY=your-key-here"
        )
    
    # Detect endpoint (OpenRouter vs OpenAI)
    if api_key.startswith("sk-or-v1-"):
        print("\nUsing OpenRouter embeddings...")
        base_url = "https://openrouter.ai/api/v1"
        model = "openai/text-embedding-3-large"
        max_tokens = 8000
    else:
        print("\nUsing OpenAI embeddings...")
        base_url = None
        model = "text-embedding-3-large"  # 3072 dimensions, 8K context
        max_tokens = 8000
    
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    
    print(f"Model: {model} (max {max_tokens} tokens)")
    print("Generating embeddings via API...")
    
    embeddings = []
    batch_size = 50  # OpenAI allows larger batches
    
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding with OpenAI"):
        batch = chunks[i:i+batch_size]
        
        # Prepare texts with source code
        texts = [_prepare_embedding_text(chunk) for chunk in batch]
        
        # Truncate to max tokens (approximate - 1 token ≈ 4 chars)
        texts = [text[:max_tokens * 4] for text in texts]
        
        try:
            response = client.embeddings.create(
                input=texts,
                model=model
            )
            
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
            
        except Exception as e:
            print(f"\n⚠️  Error in batch {i}: {e}")
            # Retry with smaller batch
            for text in texts:
                try:
                    response = client.embeddings.create(input=[text], model=model)
                    embeddings.append(response.data[0].embedding)
                except Exception as retry_error:
                    print(f"Failed to embed chunk: {retry_error}")
                    # Add zero vector as fallback
                    embeddings.append([0.0] * 3072)
    
    return np.array(embeddings)


def _get_component_stats(chunks: List[Dict]) -> Dict:
    """Get statistics on component types"""
    stats = {}
    for chunk in chunks:
        comp_type = chunk.get('spring_component_type', 'unknown')
        stats[comp_type] = stats.get(comp_type, 0) + 1
    return stats


def search_similar_code(query: str, index_dir: str, top_k: int = 5, use_openai: bool = True) -> List[Dict]:
    """
    Search for similar code chunks using semantic similarity
    
    Args:
        query: Natural language or code query
        index_dir: Directory containing FAISS index
        top_k: Number of results to return
        use_openai: Use same embedding method as indexing
    
    Returns:
        List of similar chunks with scores
    """
    
    # Load index and metadata
    index = faiss.read_index(str(Path(index_dir) / "code_index.faiss"))
    
    try:
        with open(Path(index_dir) / "metadata.json", 'r') as f:
            chunks = json.load(f)
    except FileNotFoundError:
        # Fallback to numpy
        chunks = np.load(str(Path(index_dir) / "metadata.npy"), allow_pickle=True).tolist()
    
    # Embed query
    query_chunk = {'id': 'query', 'semantic_text': query}
    
    if use_openai:
        query_embedding = _embed_with_openai([query_chunk])
    else:
        query_embedding = _embed_with_graphcodebert([query_chunk])
    
    # Search
    distances, indices = index.search(query_embedding, top_k)
    
    # Prepare results
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        chunk = chunks[idx]
        results.append({
            'chunk': chunk,
            'similarity_score': 1 / (1 + distance),  # Convert distance to similarity
            'distance': float(distance)
        })
    
    return results


def main():
    """Standalone usage"""
    import sys
    
    chunks_file = sys.argv[1] if len(sys.argv) > 1 else "/Users/home/Desktop/new/CodeMind-/platform/2.chunker/chunk_Data/chunk.json"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "/Users/home/Desktop/new/CodeMind-/platform/3.embedder/vector DB"
    use_openai = sys.argv[3] != "graphcodebert" if len(sys.argv) > 3 else True  # Default to OpenAI
    
    print("=" * 70)
    print("Code-Aware Embedder with SOURCE CODE Support")
    print("=" * 70)
    print(f"Chunks file: {chunks_file}")
    print(f"Output dir:  {output_dir}")
    print(f"Model:       {'OpenAI API (8K context)' if use_openai else 'GraphCodeBERT (512 tokens)'}")
    print("=" * 70)
    print()
    
    create_codebert_embeddings(chunks_file, output_dir, use_openai)
    
    print("\n" + "=" * 70)
    print("✅ Embedding complete!")
    print("=" * 70)
    print("\nGenerated files:")
    print(f"  📊 code_index.faiss     - Vector index for similarity search")
    print(f"  📄 metadata.json        - Full chunk data with source code")
    print(f"  📈 embedding_stats.json - Embedding statistics")
    print("\nYou can now:")
    print("  1. Use the FAISS index for semantic code search")
    print("  2. Query similar methods by natural language or code")
    print("  3. Build RAG systems with source code context")
    print("=" * 70)


if __name__ == "__main__":
    main()