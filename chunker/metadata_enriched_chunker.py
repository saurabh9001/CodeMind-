"""
Metadata-Enriched Chunker - Creates rich chunks from COMPLETE_DEEP_ANALYSIS.json
Works without source code by leveraging comprehensive analysis metadata
"""
import json
from typing import List, Dict
from pathlib import Path


def create_enriched_chunks(analysis_json_path: str, output_path: str) -> List[Dict]:
    """Create semantically rich chunks from analysis JSON"""
    
    print(f"Loading analysis from {analysis_json_path}...")
    with open(analysis_json_path, 'r') as f:
        analysis = json.load(f)
    
    print("Creating enriched chunks...")
    
    # Extract all analysis data
    method_call_graph = analysis.get('method_call_graph', {})
    class_to_file = analysis.get('class_to_file_mapping', {})
    class_dependencies = analysis.get('class_dependencies', {})
    spring_components = analysis.get('spring_components', {})
    method_complexity = analysis.get('method_complexity', {})
    db_operations = analysis.get('database_operations', [])
    scheduled_tasks = analysis.get('scheduled_tasks', [])
    error_handling = analysis.get('error_handling', [])
    config_usage = analysis.get('configuration_usage', [])
    domain_models = analysis.get('domain_model_usage', {})
    api_endpoints = analysis.get('api_endpoints', [])
    
    # Index by method
    db_ops_by_method = {}
    for op in db_operations:
        key = f"{op['class']}.{op.get('method', 'unknown')}"
        if key not in db_ops_by_method:
            db_ops_by_method[key] = []
        db_ops_by_method[key].append(op)
    
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
    
    config_by_method = {}
    for cfg in config_usage:
        method = cfg.get('method', 'unknown')
        key = f"{cfg['class']}.{method}"
        if key not in config_by_method:
            config_by_method[key] = []
        config_by_method[key].append(cfg)
    
    api_by_class = {}
    for endpoint in api_endpoints:
        cls = endpoint['class']
        if cls not in api_by_class:
            api_by_class[cls] = []
        api_by_class[cls].append(endpoint)
    
    # Create enriched chunks
    chunks = []
    
    for method_full_name, graph_data in method_call_graph.items():
        if '.' not in method_full_name:
            continue
        
        class_name, method_name = method_full_name.rsplit('.', 1)
        
        # Get all available metadata
        complexity_data = method_complexity.get(method_full_name, {})
        scheduled_task = scheduled_by_method.get(method_full_name)
        db_ops = db_ops_by_method.get(method_full_name, [])
        errors = error_by_method.get(method_full_name, [])
        configs = config_by_method.get(method_full_name, [])
        
        # Build semantic description (compensates for missing source code)
        description_parts = []
        
        # Component type
        component_type = spring_components.get(class_name, 'Component')
        description_parts.append(f"Spring {component_type}")
        
        # Complexity
        if complexity_data:
            risk = complexity_data.get('risk', 'unknown')
            complexity = complexity_data.get('complexity', 0)
            description_parts.append(f"complexity={complexity} risk={risk}")
        
        # Database operations
        if db_ops:
            op_types = set(op.get('operation', op.get('type', 'unknown')) for op in db_ops)
            description_parts.append(f"DB operations: {', '.join(op_types)}")
            for op in db_ops:
                if 'entity' in op:
                    description_parts.append(f"entity={op['entity']}")
        
        # Scheduled/async
        if scheduled_task:
            task_type = scheduled_task.get('type', 'scheduled')
            description_parts.append(f"{task_type} task")
            if 'schedule' in scheduled_task:
                description_parts.append(f"schedule={scheduled_task['schedule']}")
        
        # Error handling
        if errors:
            error_types = set(e.get('type', 'unknown') for e in errors)
            description_parts.append(f"Error handling: {', '.join(error_types)}")
        
        # Config usage
        if configs:
            config_keys = [c['key'] for c in configs]
            description_parts.append(f"Uses config: {', '.join(config_keys[:3])}")
        
        # API endpoints (if class has them)
        class_endpoints = api_by_class.get(class_name, [])
        if class_endpoints:
            methods = [ep['method'] for ep in class_endpoints]
            paths = [ep['path'] for ep in class_endpoints]
            description_parts.append(f"API endpoints: {', '.join(set(methods))} {paths[:2]}")
        
        # Domain models
        models_used = domain_models.get(class_name, [])
        if models_used:
            description_parts.append(f"Uses models: {', '.join(models_used[:5])}")
        
        # Call relationships - format for better semantic understanding
        calls_formatted = []
        for call in graph_data.get('calls', []):
            target = call.get('target', '')
            call_type = call.get('type', 'unknown')
            calls_formatted.append(f"{target} ({call_type})")
        
        called_by = graph_data.get('called_by', [])
        
        # Build semantic text for embedding
        semantic_text = f"""
Method: {method_full_name}
Class: {class_name} ({component_type})
File: {class_to_file.get(class_name, 'unknown')}

Description: {' | '.join(description_parts)}

Calls ({len(calls_formatted)}):
{chr(10).join(f"  - {c}" for c in calls_formatted[:20])}

Called by ({len(called_by)}):
{chr(10).join(f"  - {c}" for c in called_by[:20])}

Dependencies: {', '.join(list(class_dependencies.get(class_name, []))[:10])}
""".strip()
        
        # Create enriched chunk
        chunk = {
            'id': method_full_name,
            'type': 'method',
            'class_name': class_name,
            'method_name': method_name,
            'file_path': class_to_file.get(class_name, ''),
            
            # Semantic text for embedding (rich description)
            'semantic_text': semantic_text,
            
            # Component metadata
            'spring_component_type': component_type,
            'component_layer': _infer_layer(component_type),
            
            # Relationships
            'calls': [c.get('target', '') for c in graph_data.get('calls', [])],
            'calls_detail': graph_data.get('calls', [])[:20],  # Limit detail
            'called_by': called_by[:20],
            'class_dependencies': list(class_dependencies.get(class_name, []))[:20],
            
            # Quality metrics
            'complexity': complexity_data.get('complexity', 0),
            'risk_level': complexity_data.get('risk', 'unknown'),
            
            # Features
            'database_operations': db_ops,
            'is_scheduled': scheduled_task is not None,
            'scheduled_info': scheduled_task if scheduled_task else {},
            'error_handling': errors,
            'config_usage': configs,
            
            # Domain context
            'domain_models': models_used[:10],
            'api_endpoints': class_endpoints[:5],
            
            # Search optimization
            'search_keywords': _extract_keywords(method_full_name, description_parts, calls_formatted, called_by)
        }
        
        chunks.append(chunk)
        
        if len(chunks) % 100 == 0:
            print(f"  Processed {len(chunks)} chunks...")
    
    print(f"\n✓ Created {len(chunks)} enriched chunks")
    
    # Save chunks
    with open(output_path, 'w') as f:
        json.dump(chunks, f, indent=2)
    
    print(f"✓ Saved to {output_path}")
    
    # Stats
    print("\nChunk Statistics:")
    print(f"  Methods with DB operations: {sum(1 for c in chunks if c['database_operations'])}")
    print(f"  Scheduled/async methods: {sum(1 for c in chunks if c['is_scheduled'])}")
    print(f"  Methods with error handling: {sum(1 for c in chunks if c['error_handling'])}")
    print(f"  High complexity (risk): {sum(1 for c in chunks if c['risk_level'] == 'high')}")
    print(f"  Medium complexity: {sum(1 for c in chunks if c['risk_level'] == 'medium')}")
    
    return chunks


def _infer_layer(component_type: str) -> str:
    """Infer architectural layer from component type"""
    if 'Controller' in component_type:
        return 'presentation'
    elif 'Service' in component_type:
        return 'business'
    elif 'Repository' in component_type or 'DAO' in component_type:
        return 'data'
    elif 'Configuration' in component_type:
        return 'infrastructure'
    return 'unknown'


def _extract_keywords(method_name: str, description_parts: List[str], calls: List[str], called_by: List[str]) -> List[str]:
    """Extract searchable keywords from method context"""
    keywords = []
    
    # Method name parts
    import re
    name_parts = re.findall(r'[A-Z][a-z]+|[a-z]+', method_name)
    keywords.extend(name_parts)
    
    # From description
    for part in description_parts:
        keywords.extend(part.split())
    
    # Call targets (method names only)
    for call in calls[:10]:
        if '.' in call:
            keywords.append(call.split('.')[-1].split('(')[0])
    
    # Caller names
    for caller in called_by[:10]:
        if '.' in caller:
            keywords.append(caller.split('.')[-1])
    
    # Deduplicate and filter
    keywords = list(set(k.lower() for k in keywords if len(k) > 3))
    return keywords[:50]


if __name__ == "__main__":
    import sys
    
    analysis_file = sys.argv[1] if len(sys.argv) > 1 else "/Users/home/Desktop/p/COMPLETE_DEEP_ANALYSIS.json"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "/Users/home/Desktop/p/data/enriched_chunks.json"
    
    chunks = create_enriched_chunks(analysis_file, output_file)
    
    print(f"\n✅ Created {len(chunks)} metadata-enriched chunks!")
    print("\nThese chunks include:")
    print("  ✓ Comprehensive semantic descriptions")
    print("  ✓ Call graphs and dependencies")
    print("  ✓ Database operations and entity mappings")
    print("  ✓ Scheduled tasks and async patterns")
    print("  ✓ Error handling strategies")
    print("  ✓ Configuration usage")
    print("  ✓ Domain models and API endpoints")
    print("  ✓ Complexity metrics and risk levels")
    print("  ✓ Search-optimized keywords")
