"""
Metadata-Enriched Chunker with SOURCE CODE - Creates rich chunks from COMPLETE_DEEP_ANALYSIS
Now includes actual source code for methods and classes
"""
import json
from typing import List, Dict
from pathlib import Path


def create_enriched_chunks(analysis_json_path: str, output_path: str) -> List[Dict]:
    """Create semantically rich chunks from analysis JSON with source code"""
    
    print(f"Loading analysis from {analysis_json_path}...")
    with open(analysis_json_path, 'r') as f:
        analysis = json.load(f)
    
    print("Creating enriched chunks with source code...")
    
    # Extract all analysis data
    methods_data = analysis.get('methods', {})
    classes_data = analysis.get('classes', {})
    
    # Legacy support - fallback to old structure if new one doesn't exist
    if not methods_data:
        methods_data = analysis.get('method_call_graph', {})
    
    class_to_file = analysis.get('class_to_file_mapping', {})
    external_apis = analysis.get('external_api_calls', [])
    db_operations = analysis.get('database_operations', [])
    scheduled_tasks = analysis.get('scheduled_tasks', [])
    error_handling = analysis.get('error_handling', [])
    config_usage = analysis.get('configuration_usage', [])
    
    # Index data by method/class
    external_apis_by_class = {}
    for api in external_apis:
        cls = api.get('class', '')
        if cls not in external_apis_by_class:
            external_apis_by_class[cls] = []
        external_apis_by_class[cls].append(api)
    
    db_ops_by_class = {}
    for op in db_operations:
        cls = op.get('class', '')
        if cls not in db_ops_by_class:
            db_ops_by_class[cls] = []
        db_ops_by_class[cls].append(op)
    
    scheduled_by_method = {}
    for task in scheduled_tasks:
        key = f"{task['class']}.{task['method']}"
        scheduled_by_method[key] = task
    
    error_by_method = {}
    for err in error_handling:
        key = f"{err['class']}.{err['method']}"
        if key not in error_by_method:
            error_by_method[key] = []
        error_by_method[key].append(err)
    
    config_by_class = {}
    for cfg in config_usage:
        cls = cfg.get('class', '')
        if cls not in config_by_class:
            config_by_class[cls] = []
        config_by_class[cls].append(cfg)
    
    # Create enriched chunks
    chunks = []
    
    for method_full_name, method_data in methods_data.items():
        if '.' not in method_full_name:
            continue
        
        class_name, method_name = method_full_name.rsplit('.', 1)
        
        # Extract source code (new structure)
        source_info = method_data.get('source', {})
        source_code = source_info.get('source_code', '')
        annotations = source_info.get('annotations', [])
        signature = source_info.get('signature', '')
        start_line = source_info.get('start_line', 0)
        end_line = source_info.get('end_line', 0)
        lines_of_code = source_info.get('lines_of_code', 0)
        
        # Get class-level info
        class_info = classes_data.get(class_name, {})
        class_source = class_info.get('source', {})
        component_type = class_info.get('component_type', 'unknown')
        class_dependencies = class_info.get('dependencies', [])
        class_package = class_source.get('package', '')
        class_imports = class_source.get('imports', [])
        class_annotations = class_source.get('annotations', [])
        
        # Get call graph data
        calls = method_data.get('calls', [])
        called_by = method_data.get('called_by', [])
        
        # Get metadata
        scheduled_task = scheduled_by_method.get(method_full_name)
        db_ops = db_ops_by_class.get(class_name, [])
        errors = error_by_method.get(method_full_name, [])
        configs = config_by_class.get(class_name, [])
        apis = external_apis_by_class.get(class_name, [])
        
        # Build semantic description
        description_parts = []
        
        # Component type
        description_parts.append(f"Spring {component_type}")
        
        # Source code info
        if lines_of_code:
            description_parts.append(f"{lines_of_code} lines of code")
        
        # Annotations
        if annotations:
            description_parts.append(f"Annotations: {', '.join(annotations)}")
        
        # Database operations
        if db_ops:
            op_types = set(op.get('type', 'unknown') for op in db_ops)
            description_parts.append(f"DB operations: {', '.join(op_types)}")
        
        # Scheduled/async
        if scheduled_task:
            task_type = scheduled_task.get('type', 'scheduled')
            description_parts.append(f"{task_type} task")
            if 'config' in scheduled_task:
                config = scheduled_task['config']
                if 'cron' in config:
                    description_parts.append(f"cron={config['cron']}")
        
        # Error handling
        if errors:
            exception_types = set(e.get('exception_type', 'unknown') for e in errors)
            description_parts.append(f"Handles: {', '.join(exception_types)}")
        
        # Config usage
        if configs:
            config_props = [c.get('property', '') for c in configs]
            description_parts.append(f"Config: {', '.join(config_props[:3])}")
        
        # API calls
        if apis:
            api_types = set(api.get('client_type', 'unknown') for api in apis)
            description_parts.append(f"External APIs: {', '.join(api_types)}")
        
        # Call relationships - format for better semantic understanding
        calls_formatted = []
        for call in calls:
            target = call.get('target', '') if isinstance(call, dict) else str(call)
            call_type = call.get('type', 'unknown') if isinstance(call, dict) else 'unknown'
            calls_formatted.append(f"{target} ({call_type})")
        
        # Build semantic text for embedding (includes source code context)
        semantic_text_parts = [
            f"Method: {method_full_name}",
            f"Class: {class_name} ({component_type})",
            f"Package: {class_package}",
            f"File: {class_to_file.get(class_name, 'unknown')}",
            "",
            f"Description: {' | '.join(description_parts)}",
        ]
        
        # Add signature if available
        if signature:
            semantic_text_parts.append(f"\nSignature:\n{signature}")
        
        # Add source code snippet (first 20 lines for embedding)
        if source_code:
            code_lines = source_code.split('\n')[:20]
            semantic_text_parts.append(f"\nSource Code Preview:\n{chr(10).join(code_lines)}")
        
        # Add call information
        semantic_text_parts.extend([
            f"\nCalls ({len(calls_formatted)}):",
            chr(10).join(f"  - {c}" for c in calls_formatted[:15]),
            f"\nCalled by ({len(called_by)}):",
            chr(10).join(f"  - {c}" for c in called_by[:15]),
            f"\nDependencies: {', '.join(class_dependencies[:10])}"
        ])
        
        semantic_text = '\n'.join(semantic_text_parts)
        
        # Create enriched chunk with source code
        chunk = {
            'id': method_full_name,
            'type': 'method',
            'class_name': class_name,
            'method_name': method_name,
            'file_path': class_to_file.get(class_name, ''),
            
            # SOURCE CODE (NEW!)
            'source_code': source_code,
            'source_signature': signature,
            'source_annotations': annotations,
            'source_start_line': start_line,
            'source_end_line': end_line,
            'lines_of_code': lines_of_code,
            
            # CLASS CONTEXT (NEW!)
            'class_package': class_package,
            'class_imports': class_imports[:20],  # Limit imports
            'class_annotations': class_annotations,
            
            # Semantic text for embedding (rich description + code)
            'semantic_text': semantic_text,
            
            # Component metadata
            'spring_component_type': component_type,
            'component_layer': _infer_layer(component_type),
            
            # Relationships
            'calls': [c.get('target', '') if isinstance(c, dict) else str(c) for c in calls],
            'calls_detail': calls[:15],  # Limit detail
            'called_by': called_by[:15],
            'class_dependencies': class_dependencies[:15],
            
            # Features
            'database_operations': db_ops[:10],
            'external_api_calls': apis[:10],
            'is_scheduled': scheduled_task is not None,
            'scheduled_info': scheduled_task if scheduled_task else {},
            'error_handling': errors[:10],
            'config_usage': configs[:10],
            
            # Search optimization
            'search_keywords': _extract_keywords(
                method_full_name, 
                description_parts, 
                calls_formatted, 
                called_by,
                source_code
            )
        }
        
        chunks.append(chunk)
        
        if len(chunks) % 100 == 0:
            print(f"  Processed {len(chunks)} chunks...")
    
    print(f"\n✓ Created {len(chunks)} enriched chunks with source code")
    
    # Save chunks
    with open(output_path, 'w') as f:
        json.dump(chunks, f, indent=2)
    
    print(f"✓ Saved to {output_path}")
    
    # Stats
    print("\nChunk Statistics:")
    print(f"  Total methods: {len(chunks)}")
    print(f"  Methods with source code: {sum(1 for c in chunks if c['source_code'])}")
    print(f"  Methods with DB operations: {sum(1 for c in chunks if c['database_operations'])}")
    print(f"  Scheduled/async methods: {sum(1 for c in chunks if c['is_scheduled'])}")
    print(f"  Methods with error handling: {sum(1 for c in chunks if c['error_handling'])}")
    print(f"  Methods calling external APIs: {sum(1 for c in chunks if c['external_api_calls'])}")
    print(f"  Average LOC per method: {sum(c['lines_of_code'] for c in chunks) / len(chunks):.1f}")
    
    # Component breakdown
    component_counts = {}
    for chunk in chunks:
        comp = chunk['spring_component_type']
        component_counts[comp] = component_counts.get(comp, 0) + 1
    
    print("\nComponent Type Breakdown:")
    for comp, count in sorted(component_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {comp}: {count}")
    
    return chunks


def _infer_layer(component_type: str) -> str:
    """Infer architectural layer from component type"""
    comp_lower = component_type.lower()
    if 'controller' in comp_lower or 'rest' in comp_lower:
        return 'presentation'
    elif 'service' in comp_lower:
        return 'business'
    elif 'repository' in comp_lower or 'dao' in comp_lower:
        return 'data'
    elif 'configuration' in comp_lower or 'config' in comp_lower:
        return 'infrastructure'
    return 'unknown'


def _extract_keywords(method_name: str, description_parts: List[str], 
                      calls: List[str], called_by: List[str], 
                      source_code: str = '') -> List[str]:
    """Extract searchable keywords from method context including source code"""
    keywords = []
    
    # Method name parts (camelCase splitting)
    import re
    name_parts = re.findall(r'[A-Z][a-z]+|[a-z]+', method_name)
    keywords.extend(name_parts)
    
    # From description
    for part in description_parts:
        keywords.extend(part.split())
    
    # Call targets (method names only)
    for call in calls[:15]:
        if '.' in call:
            method_part = call.split('.')[-1].split('(')[0]
            keywords.append(method_part)
    
    # Caller names
    for caller in called_by[:15]:
        if '.' in caller:
            keywords.append(caller.split('.')[-1])
    
    # Extract keywords from source code
    if source_code:
        # Extract variable names
        var_names = re.findall(r'\b([a-z][a-zA-Z0-9]{3,})\b', source_code)
        keywords.extend(var_names[:30])
        
        # Extract class names used
        class_names = re.findall(r'\b([A-Z][a-zA-Z0-9]+)\b', source_code)
        keywords.extend(class_names[:30])
        
        # Extract string literals (potential domain terms)
        string_literals = re.findall(r'["\']([a-zA-Z][a-zA-Z0-9 ]{3,})["\']', source_code)
        for literal in string_literals[:20]:
            keywords.extend(literal.split())
    
    # Deduplicate and filter
    keywords = list(set(k.lower() for k in keywords if len(k) > 3 and k.isalnum()))
    return keywords[:100]


if __name__ == "__main__":
    import sys
    
    analysis_file = sys.argv[1] if len(sys.argv) > 1 else "/Users/home/Desktop/new/CodeMind-/platform/1.parser/output parser/fc2.json"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "/Users/home/Desktop/new/CodeMind-/platform/2.chunker/chunk_Data/chunk.json"
    
    chunks = create_enriched_chunks(analysis_file, output_file)
    
    print(f"\n✅ Created {len(chunks)} metadata-enriched chunks with SOURCE CODE!")
    print("\nThese chunks include:")
    print("  ✓ Full method source code")
    print("  ✓ Method signatures and annotations")
    print("  ✓ Class package, imports, and annotations")
    print("  ✓ Line numbers and LOC metrics")
    print("  ✓ Comprehensive semantic descriptions")
    print("  ✓ Call graphs and dependencies")
    print("  ✓ Database operations and entity mappings")
    print("  ✓ Scheduled tasks and async patterns")
    print("  ✓ Error handling strategies")
    print("  ✓ Configuration usage")
    print("  ✓ External API calls")
    print("  ✓ Search-optimized keywords (from code + metadata)")
    print("\n🎯 Ready for embedding and semantic search!")