"""
Understands code semantics, data flow, and relationships
"""
import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm


def create_codebert_embeddings(chunks_path: str, output_dir: str, use_openai: bool = False):
    """
    Create code-aware embeddings using GraphCodeBERT or OpenAI
    
    Args:
        chunks_path: Path to enriched_chunks.json
        output_dir: Directory to save FAISS index and metadata
        use_openai: If True, use OpenAI embeddings API (simpler, no local model)
    """
    
    print(f"Loading chunks from {chunks_path}...")
    with open(chunks_path, 'r') as f:
        chunks = json.load(f)
    
    print(f"Loaded {len(chunks)} chunks")
    
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
    np.save(str(output_path / "metadata.npy"), np.array(chunks, dtype=object))
    
    print(f"\n✅ Created FAISS index with {len(chunks)} vectors (dimension={dimension})")
    print(f"✓ Saved to {output_dir}/")
    
    return index, chunks


def _embed_with_graphcodebert(chunks: List[Dict]) -> np.ndarray:
    """Embed using GraphCodeBERT (Microsoft's code-aware model)"""
    
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
    
    batch_size = 16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding"):
        batch = chunks[i:i+batch_size]
        texts = [chunk['semantic_text'] for chunk in batch]
        
        # Tokenize with truncation
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
    """Embed using OpenAI embeddings API (simpler alternative)"""
    
    try:
        import openai
        from dotenv import load_dotenv
        import os
    except ImportError:
        print("\n⚠️  openai or python-dotenv not installed")
        raise
    
    # Load API key
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env")
    
    # Detect endpoint (OpenRouter vs OpenAI)
    if api_key.startswith("sk-or-v1-"):
        print("\nUsing OpenRouter embeddings...")
        base_url = "https://openrouter.ai/api/v1"
        model = "openai/text-embedding-3-large"
    else:
        print("\nUsing OpenAI embeddings...")
        base_url = None
        model = "text-embedding-3-large"
    
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    
    print("Generating embeddings via API...")
    embeddings = []
    
    batch_size = 100  # API can handle larger batches
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding"):
        batch = chunks[i:i+batch_size]
        texts = [chunk['semantic_text'][:8000] for chunk in batch]  # Truncate to 8K tokens
        
        response = client.embeddings.create(
            input=texts,
            model=model
        )
        
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)
    
    return np.array(embeddings)


def main():
    """Standalone usage"""
    import sys
    
    chunks_file = sys.argv[1] if len(sys.argv) > 1 else "/Users/home/Desktop/new/CodeMind-/platform/2.chunker/chunk_Data/chunk.json"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "/Users/home/Desktop/new/CodeMind-/platform/3.embedder/vector DB"
    use_openai = sys.argv[3] == "openai" if len(sys.argv) > 3 else True  # Default to OpenAI to avoid segfaults
    
    print("Code-Aware Embedder")
    print("=" * 50)
    print(f"Chunks: {chunks_file}")
    print(f"Output: {output_dir}")
    print(f"Model: {'OpenAI API' if use_openai else 'GraphCodeBERT (local)'}")
    print()
    
    create_codebert_embeddings(chunks_file, output_dir, use_openai)


if __name__ == "__main__":
    main()
