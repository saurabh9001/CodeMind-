"""
Raw Retriever - Returns Retrieved Context Without LLM
Get the raw chunks and metadata that would be sent to the LLM
"""
import os
import faiss
import numpy as np
import openai
from pathlib import Path
from dotenv import load_dotenv
import json


# Load environment
REPO_ROOT = Path(__file__).parent.parent.parent
load_dotenv(dotenv_path=REPO_ROOT / ".env")


def load_index_and_metadata(index_path="3.embedder/vector DB/code_index.faiss", metadata_path="3.embedder/vector DB/metadata.npy"):
    """Load FAISS index and enriched metadata"""
    base_dir = REPO_ROOT
    
    index = faiss.read_index(str(base_dir / "platform" / index_path))
    metadata = np.load(str(base_dir / "platform" / metadata_path), allow_pickle=True)
    
   
    
    return index, metadata


def retrieve_top_k(index, metadata, query, k=5, use_openai_embed=True):
    """
    Retrieve top-k relevant chunks
    
    Args:
        use_openai_embed: If True, use OpenAI embeddings (matches index if created with OpenAI)
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
        from sentence_transformers import SentenceTransformer
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


def build_llm_prompt(query, results):
    """Build the complete LLM API call structure (system + user messages)"""
    
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
    
    # System message
    system_message = "You are a precise code analyst. Give SHORT, structured answers. Use the exact format requested. NO fluff, NO introductions. Start directly with the answer. Use backticks for code elements. Maximum 4 sentences."
    
    # User message
    user_message = f"""You are an expert Java code analyst. Answer the question using ONLY the provided code context.

Retrieved Code Context:
{context}

Question: {query}

CRITICAL RULES:
1. Keep answer to 2-4 sentences maximum
2. Start with the main answer immediately - NO introductions like "Based on the code..." or "The retrieved context shows..."
3. Use this exact format:

**Answer:**
[Direct answer in 1-2 sentences]

**Location:**
- File: `[exact file path]`
- Method: `[ClassName.methodName]`

**Technical Details:**
- [Key point 1]
- [Key point 2]
- [Only if relevant: complexity/risk/DB operations]

IMPORTANT:
- Be extremely concise and technical
- Reference methods/classes in backticks
- Include exact file paths
- Skip obvious information
- If context is insufficient, state it briefly

Your response:"""
    
    # Build complete API call structure
    complete_prompt = f"""
{'='*80}
COMPLETE LLM API CALL STRUCTURE
{'='*80}

API ENDPOINT: OpenAI Chat Completions
MODEL: gpt-3.5-turbo (or openai/gpt-3.5-turbo via OpenRouter)
MAX_TOKENS: 500
TEMPERATURE: 0.1

{'='*80}
MESSAGE 1: SYSTEM
{'='*80}
{system_message}

{'='*80}
MESSAGE 2: USER
{'='*80}
{user_message}

{'='*80}
END OF PROMPT
{'='*80}

SUMMARY:
- Total chunks retrieved: {len(results)}
- System message length: {len(system_message)} chars
- User message length: {len(user_message)} chars
- Total prompt length: ~{len(system_message) + len(user_message)} chars
"""
    
    return complete_prompt


def format_retrieved_context(query, results, output_format='text'):
    """
    Format retrieved context in different formats
    
    Args:
        query: The search query
        results: List of retrieved chunks with distances
        output_format: 'text', 'json', 'detailed', or 'prompt'
    
    Returns:
        Formatted string or dict
    """
    
    if output_format == 'prompt':
        # Return the complete LLM prompt
        return build_llm_prompt(query, results)
    
    elif output_format == 'json':
        # Return as JSON structure
        output = {
            'query': query,
            'total_results': len(results),
            'chunks': []
        }
        
        for r in results:
            chunk = r['chunk']
            output['chunks'].append({
                'rank': r['rank'],
                'distance': r['distance'],
                'id': chunk['id'],
                'file_path': chunk.get('file_path', 'unknown'),
                'spring_component_type': chunk.get('spring_component_type', 'unknown'),
                'component_layer': chunk.get('component_layer', 'unknown'),
                'complexity': chunk.get('complexity', 0),
                'risk_level': chunk.get('risk_level', 'unknown'),
                'semantic_text': chunk.get('semantic_text', ''),
                'calls': chunk.get('calls', []),
                'called_by': chunk.get('called_by', []),
                'class_dependencies': chunk.get('class_dependencies', []),
                'database_operations': chunk.get('database_operations', []),
                'is_scheduled': chunk.get('is_scheduled', False),
                'scheduled_info': chunk.get('scheduled_info', {}),
                'error_handling': chunk.get('error_handling', []),
                'domain_models_used': chunk.get('domain_models_used', []),
                'config_properties': chunk.get('config_properties', [])
            })
        
        return json.dumps(output, indent=2)
    
    elif output_format == 'detailed':
        # Detailed text format with all metadata
        lines = []
        lines.append("=" * 80)
        lines.append(f"QUERY: {query}")
        lines.append(f"RETRIEVED: {len(results)} chunks")
        lines.append("=" * 80)
        lines.append("")
        
        for r in results:
            chunk = r['chunk']
            
            lines.append(f"{'='*80}")
            lines.append(f"RANK #{r['rank']} - Distance: {r['distance']:.4f}")
            lines.append(f"{'='*80}")
            lines.append(f"ID: {chunk['id']}")
            lines.append(f"Type: {chunk.get('spring_component_type', 'unknown')} ({chunk.get('component_layer', 'unknown')} layer)")
            lines.append(f"File: {chunk.get('file_path', 'unknown')}")
            lines.append(f"Complexity: {chunk.get('complexity', 0)} | Risk: {chunk.get('risk_level', 'unknown')}")
            lines.append("")
            
            lines.append("SEMANTIC DESCRIPTION:")
            lines.append("-" * 80)
            lines.append(chunk.get('semantic_text', 'No description'))
            lines.append("")
            
            lines.append("CALL RELATIONSHIPS:")
            lines.append(f"  • Calls {len(chunk.get('calls', []))} methods:")
            for call in chunk.get('calls', [])[:10]:
                lines.append(f"    - {call}")
            if len(chunk.get('calls', [])) > 10:
                lines.append(f"    ... and {len(chunk.get('calls', [])) - 10} more")
            lines.append("")
            
            lines.append(f"  • Called by {len(chunk.get('called_by', []))} methods:")
            for caller in chunk.get('called_by', [])[:5]:
                lines.append(f"    - {caller}")
            if len(chunk.get('called_by', [])) > 5:
                lines.append(f"    ... and {len(chunk.get('called_by', [])) - 5} more")
            lines.append("")
            
            if chunk.get('class_dependencies'):
                lines.append("DEPENDENCIES:")
                for dep in chunk.get('class_dependencies', [])[:10]:
                    lines.append(f"  - {dep}")
                lines.append("")
            
            if chunk.get('database_operations'):
                lines.append("DATABASE OPERATIONS:")
                for op in chunk.get('database_operations', []):
                    lines.append(f"  - {op}")
                lines.append("")
            
            if chunk.get('is_scheduled'):
                lines.append("SCHEDULED TASK INFO:")
                sched_info = chunk.get('scheduled_info', {})
                for key, value in sched_info.items():
                    lines.append(f"  - {key}: {value}")
                lines.append("")
            
            if chunk.get('error_handling'):
                lines.append("ERROR HANDLING:")
                for eh in chunk.get('error_handling', []):
                    lines.append(f"  - {eh}")
                lines.append("")
            
            if chunk.get('domain_models_used'):
                lines.append(f"DOMAIN MODELS: {', '.join(chunk.get('domain_models_used', []))}")
                lines.append("")
            
            if chunk.get('config_properties'):
                lines.append("CONFIGURATION PROPERTIES:")
                for prop in chunk.get('config_properties', []):
                    lines.append(f"  - {prop}")
                lines.append("")
            
            lines.append("")
        
        return "\n".join(lines)
    
    else:  # 'text' format (default - shows LLM prompt structure)
        lines = []
        lines.append("=" * 80)
        lines.append("COMPLETE LLM PROMPT STRUCTURE")
        lines.append("=" * 80)
        lines.append("")
        
        # System prompt
        lines.append("SYSTEM PROMPT:")
        lines.append("-" * 80)
        system_msg = "You are a precise code analyst. Give SHORT, structured answers. Use the exact format requested. NO fluff, NO introductions. Start directly with the answer. Use backticks for code elements. Maximum 4 sentences."
        lines.append(system_msg)
        lines.append("")
        lines.append("=" * 80)
        lines.append("USER PROMPT:")
        lines.append("=" * 80)
        lines.append("")
        lines.append("You are an expert Java code analyst. Answer the question using ONLY the provided code context.")
        lines.append("")
        lines.append(f"QUERY: {query}")
        lines.append("")
        lines.append(f"RETRIEVED CONTEXT ({len(results)} chunks):")
        lines.append("-" * 80)
        
        for r in results:
            chunk = r['chunk']
            
            lines.append(f"\n--- CHUNK {r['rank']}: {chunk['id']} (distance={r['distance']:.4f}) ---")
            lines.append(f"Type: {chunk.get('spring_component_type', 'unknown')} ({chunk.get('component_layer', 'unknown')} layer)")
            lines.append(f"File: {chunk.get('file_path', 'unknown')}")
            lines.append(f"Complexity: {chunk.get('complexity', 0)} | Risk: {chunk.get('risk_level', 'unknown')}")
            lines.append("")
            lines.append(chunk.get('semantic_text', ''))
            lines.append("")
            lines.append("Metadata:")
            lines.append(f"  - Calls {len(chunk.get('calls', []))} methods: {', '.join(chunk.get('calls', [])[:10])}")
            lines.append(f"  - Called by {len(chunk.get('called_by', []))} methods: {', '.join(chunk.get('called_by', [])[:5])}")
            lines.append(f"  - Dependencies: {', '.join(chunk.get('class_dependencies', [])[:5])}")
            lines.append(f"  - DB Operations: {len(chunk.get('database_operations', []))}")
            lines.append(f"  - Scheduled: {chunk.get('is_scheduled', False)}")
            lines.append(f"  - Error Handling: {len(chunk.get('error_handling', []))} patterns")
            
            if chunk.get('database_operations'):
                lines.append("\n  Database Operations:")
                for op in chunk.get('database_operations', [])[:3]:
                    lines.append(f"    - {op}")
            
            if chunk.get('is_scheduled'):
                lines.append(f"\n  Scheduled Info: {chunk.get('scheduled_info', {})}")
        
        lines.append("")
        lines.append("=" * 80)
        lines.append("RESPONSE FORMAT INSTRUCTIONS:")
        lines.append("-" * 80)
        lines.append("**Answer:** [Direct answer in 1-2 sentences]")
        lines.append("**Location:** File: `[path]` | Method: `[ClassName.methodName]`")
        lines.append("**Technical Details:** [Key points about implementation]")
        lines.append("")
        lines.append("=" * 80)
        lines.append(f"SUMMARY: {len(results)} chunks | ~{sum(len(chunk.get('semantic_text', '')) for chunk in [r['chunk'] for r in results])} chars")
        lines.append("=" * 80)
        
        return "\n".join(lines)


def interactive_loop():
    
    
    # Load index
    index, metadata = load_index_and_metadata()
 
    print("Anlyze this and answer the following question with refernce given")
    print()
    
    output_format = 'text'
    top_k = 5
    
    while True:
        user_input = input(">>> ").strip()
        
        if not user_input:
            continue
        
        # Parse commands
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        elif user_input.lower().startswith('format '):
            new_format = user_input[7:].strip()
            if new_format in ['text', 'prompt', 'detailed', 'json']:
                output_format = new_format
                print(f"✓ Output format set to: {output_format}")
            else:
                print("Error: Format must be 'text', 'prompt', 'detailed', or 'json'")
            continue
        
        elif user_input.lower().startswith('top '):
            try:
                top_k = int(user_input[4:].strip())
                print(f"✓ Top K set to: {top_k}")
            except ValueError:
                print("Error: Top K must be a number")
            continue
        
        # Treat as query
        query = user_input
        
        print(f"\nSearching for: '{query}'")
        print(f"Format: {output_format} | Top: {top_k}\n")
        
        # Retrieve relevant chunks
        results = retrieve_top_k(index, metadata, query, k=top_k, use_openai_embed=True)
        
        # Format and display
        output = format_retrieved_context(query, results, output_format=output_format)
        print(output)
        print("\n")


if __name__ == "__main__":
    interactive_loop()
