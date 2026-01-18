import os
import json
from pathlib import Path
import faiss
import numpy as np
import textwrap
try:
    import openai
except Exception:
    openai = None

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from sentence_transformers import SentenceTransformer


REPO_ROOT = Path(__file__).resolve().parents[1]

# Load .env from repository root
if load_dotenv is not None:
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
INDEX_PATH = REPO_ROOT / "vector_db" / "code_index.faiss"
METADATA_PATH = REPO_ROOT / "vector_db" / "metadata.npy"


def load_index_and_metadata(index_path=INDEX_PATH, metadata_path=METADATA_PATH):
    index = faiss.read_index(str(index_path))
    metadata = np.load(str(metadata_path), allow_pickle=True)
    return index, metadata


def build_prompt(query: str, chunks: list) -> str:
    header = (
        "You are an assistant that answers questions about the provided code snippets. "
        "Answer concisely and only using the information available in the snippets. "
        "If the answer cannot be determined from the snippets, say you don't know.\n\n"
    )

    ctx = []
    for i, c in enumerate(chunks, start=1):
        method = c.get("method") or c.get("name") or "<unknown>"
        # Try multiple possible keys for code content
        snippet = (
            c.get("text") or 
            c.get("code") or 
            c.get("content") or 
            c.get("snippet") or 
            c.get("source") or
            c.get("body") or
            str(c.get("calls", "")) or  # fallback: show calls if no code
            ""
        )
        
        # If still empty, show all available keys for debugging
        if not snippet or len(snippet.strip()) == 0:
            snippet = f"[No code content found. Available fields: {', '.join(c.keys())}]\n"
            # Show first few fields
            for k, v in list(c.items())[:5]:
                snippet += f"{k}: {str(v)[:200]}\n"
        
        file_info = c.get("file") or c.get("path") or c.get("filename") or ""
        header_line = f"--- CHUNK {i}: {method}"
        if file_info:
            header_line += f" ({file_info})"
        header_line += " ---\n"
        
        piece = header_line + snippet + "\n"
        ctx.append(piece)

    context = "\n".join(ctx)

    prompt = header + "Context:\n" + context + "\nUser question: " + query
    return prompt


def call_openai(prompt: str, model: str = "gpt-3.5-turbo") -> str:
    if openai is None:
        raise RuntimeError("openai package not installed. Install with `pip install openai`.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    # Detect OpenRouter API key (starts with sk-or-v1-)
    if api_key.startswith("sk-or-v1-"):
        base_url = "https://openrouter.ai/api/v1"
        # Use a more capable model for OpenRouter
        if model == "gpt-3.5-turbo":
            model = "openai/gpt-3.5-turbo"
    else:
        base_url = None  # Use default OpenAI endpoint

    # OpenAI v1.0+ API
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    messages = [
        {"role": "system", "content": "You are a helpful assistant for reading code snippets."},
        {"role": "user", "content": prompt},
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=512,
        temperature=0.2,
    )

    return resp.choices[0].message.content.strip()


def retrieve_top_k(index, metadata, model, query: str, k: int = 5):
    qvec = model.encode([query])
    distances, indices = index.search(qvec, k=k)
    chunks = []
    for idx in indices[0]:
        try:
            chunk = metadata[idx].item() if hasattr(metadata[idx], "item") else metadata[idx]
        except Exception:
            chunk = metadata[idx]
        # ensure it's a dict
        if isinstance(chunk, (list, tuple)):
            # some metadata formats may store tuples
            try:
                chunk = dict(chunk)
            except Exception:
                chunk = {"method": str(chunk)}

        chunks.append(chunk)
    return chunks, distances[0]


def interactive_loop(index, metadata, embedder_model):
    print("Code QA ready. Ask questions about the code (Ctrl-C to exit).")
    while True:
        try:
            query = input("\nYour question: ")
            if not query.strip():
                continue

            chunks, dists = retrieve_top_k(index, metadata, embedder_model, query, k=5)
            prompt = build_prompt(query, chunks)

            print("\nTop retrieved chunks (brief):")
            for i, c in enumerate(chunks, start=1):
                name = c.get("method") or c.get("name") or "<unknown>"
                print(f"{i}. {name} (dist={dists[i-1]:.4f})")

            if os.getenv("OPENAI_API_KEY") and openai is not None:
                try:
                    answer = call_openai(prompt)
                    print("\nLLM Answer:\n")
                    print(textwrap.indent(answer, "  "))
                except Exception as e:
                    print("\nOpenAI call failed:", e)
                    print("Falling back to showing retrieved context only.\n")
                    print(prompt)
            else:
                print("\nOPENAI_API_KEY not set or openai package missing.")
                print("Showing retrieved context instead:\n")
                print(prompt)

        except KeyboardInterrupt:
            print("\nExiting.")
            break


def main():
    # Debug: Check if API key is loaded
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print(f"✓ API key loaded (preview: {api_key[:20]}...)")
    else:
        print("⚠ No API key found. Set OPENAI_API_KEY in .env file")
    
    index, metadata = load_index_and_metadata()
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    interactive_loop(index, metadata, embedder)


if __name__ == "__main__":
    main()
