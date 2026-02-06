"""
COMPLETE Deep Code Analysis with SOURCE CODE EXTRACTION
Extracts: Call graphs, dependencies, patterns + ACTUAL SOURCE CODE
"""
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from collections import defaultdict
import json
import re
import tree_sitter_java as tsjava
from tree_sitter import Language, Parser

console = Console()

DEFAULT_REPO_PATH = "/Users/home/Desktop/new/CodeMind-/project/bahmni-core-master/bahmnicore-ui"
DEFAULT_OUTPUT_FILE = "/Users/home/Desktop/new/CodeMind-/platform/1.parser/output parser/fc2.json"

class ParsedMethod:
    def __init__(self, name, start_line, end_line):
        self.name = name
        self.start_line = start_line
        self.end_line = end_line

class ParsedClass:
    def __init__(self, name):
        self.name = name
        self.methods = []

class ParsedFile:
    def __init__(self, file_path):
        self.file_path = file_path
        self.classes = []

def parse_java_file(file_path):
    """Parse a Java file using tree-sitter"""
    try:
        JAVA_LANGUAGE = Language(tsjava.language())
        parser = Parser(JAVA_LANGUAGE)
        
        with open(file_path, 'rb') as f:
            content = f.read()
        
        tree = parser.parse(content)
        root = tree.root_node
        
        parsed_file = ParsedFile(file_path)
        
        def traverse(node):
            if node.type == 'class_declaration':
                class_name = None
                for child in node.children:
                    if child.type == 'identifier':
                        class_name = child.text.decode('utf8')
                        break
                
                if class_name:
                    parsed_class = ParsedClass(class_name)
                    
                    body = None
                    for child in node.children:
                        if child.type == 'class_body':
                            body = child
                            break
                    
                    if body:
                        for child in body.children:
                            if child.type == 'method_declaration':
                                method_name = None
                                for mchild in child.children:
                                    if mchild.type == 'identifier':
                                        method_name = mchild.text.decode('utf8')
                                        break
                                
                                if method_name:
                                    method = ParsedMethod(
                                        method_name,
                                        child.start_point[0] + 1,
                                        child.end_point[0] + 1
                                    )
                                    parsed_class.methods.append(method)
                    
                    parsed_file.classes.append(parsed_class)
            
            for child in node.children:
                traverse(child)
        
        traverse(root)
        return parsed_file
    except Exception as e:
        return ParsedFile(file_path)

def parse_repository(repo_path):
    """Parse all Java files in repository"""
    parsed_files = []
    repo_path = Path(repo_path)
    
    for java_file in repo_path.rglob('*.java'):
        if '/test/' not in str(java_file):
            parsed = parse_java_file(java_file)
            if parsed.classes:
                parsed_files.append(parsed)
    
    return parsed_files

class CompleteAnalyzer:
    def __init__(self):
        self.method_calls = {}
        self.class_dependencies = defaultdict(set)
        self.external_apis = []
        self.database_operations = []
        self.scheduled_tasks = []
        self.spring_components = {}
        self.model_usage = defaultdict(list)
        self.error_handling = []
        self.config_usage = []
        
        # NEW: Source code storage
        self.method_source_code = {}  # method_full_name -> {source_code, annotations, signature}
        self.class_source_code = {}   # class_name -> {source_code, annotations, package}
        
        self.class_to_file = {}
        self.method_to_class = {}
        self.file_contents = {}
    
    def read_file(self, file_path):
        """Read and cache file content"""
        if file_path not in self.file_contents:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.file_contents[file_path] = f.read()
            except:
                self.file_contents[file_path] = ""
        return self.file_contents[file_path]
    
    def extract_method_source_code(self, file_path, method_name, start_line, end_line):
        """Extract the actual source code of a method with annotations"""
        content = self.read_file(file_path)
        lines = content.split('\n')
        
        # Look back for annotations (up to 10 lines before method)
        annotation_start = max(0, start_line - 10)
        pre_method_lines = lines[annotation_start:start_line-1]
        
        annotations = []
        for line in reversed(pre_method_lines):
            stripped = line.strip()
            if stripped.startswith('@'):
                annotations.insert(0, stripped)
            elif stripped and not stripped.startswith('//'):
                break
        
        # Extract method signature
        method_signature = lines[start_line-1].strip()
        
        # Extract full method body
        method_body = '\n'.join(lines[start_line-1:end_line])
        
        return {
            'source_code': method_body,
            'annotations': annotations,
            'signature': method_signature,
            'start_line': start_line,
            'end_line': end_line,
            'lines_of_code': end_line - start_line + 1
        }
    
    def extract_class_source_code(self, file_path, class_name):
        """Extract the full source code of a class with package and imports"""
        content = self.read_file(file_path)
        
        # Extract package
        package_match = re.search(r'package\s+([\w.]+);', content)
        package = package_match.group(1) if package_match else ""
        
        # Extract imports
        imports = re.findall(r'import\s+([\w.]+);', content)
        
        # Extract class annotations
        class_pattern = rf'(@\w+[^c]*)?(?:public|private|protected)?\s*(?:abstract|final)?\s*class\s+{class_name}'
        class_match = re.search(class_pattern, content, re.DOTALL)
        
        annotations = []
        if class_match and class_match.group(1):
            annotations = [a.strip() for a in re.findall(r'@\w+(?:\([^)]*\))?', class_match.group(1))]
        
        # Find class body
        class_start = content.find(f'class {class_name}')
        if class_start != -1:
            brace_count = 0
            in_class = False
            end_pos = class_start
            
            for i, char in enumerate(content[class_start:], class_start):
                if char == '{':
                    brace_count += 1
                    in_class = True
                elif char == '}':
                    brace_count -= 1
                    if in_class and brace_count == 0:
                        end_pos = i + 1
                        break
            
            class_source = content[class_start:end_pos]
        else:
            class_source = content
        
        return {
            'source_code': class_source,
            'package': package,
            'imports': imports,
            'annotations': annotations,
            'file_path': str(file_path)
        }
    
    def extract_full_method_calls(self, file_path, class_name, method_name, start_line, end_line):
        """Extract ALL method calls from a method body"""
        content = self.read_file(file_path)
        lines = content.split('\n')
        method_body = '\n'.join(lines[start_line-1:end_line])
        
        calls = []
        
        # Pattern 1: object.method()
        for match in re.finditer(r'(\w+)\.(\w+)\s*\(', method_body):
            obj, method = match.group(1), match.group(2)
            if method not in ['if', 'for', 'while', 'switch', 'return']:
                calls.append({
                    'target': f"{obj}.{method}",
                    'object': obj,
                    'method': method,
                    'type': 'instance_call'
                })
        
        # Pattern 2: ClassName.staticMethod()
        for match in re.finditer(r'([A-Z]\w+)\.(\w+)\s*\(', method_body):
            cls, method = match.group(1), match.group(2)
            calls.append({
                'target': f"{cls}.{method}",
                'class': cls,
                'method': method,
                'type': 'static_call'
            })
        
        # Pattern 3: Direct method calls (same class)
        for match in re.finditer(r'^\s*(\w+)\s*\(', method_body, re.MULTILINE):
            method = match.group(1)
            if method not in ['if', 'for', 'while', 'switch', 'catch', 'return', 'new', 'super', 'this']:
                if not method[0].isupper():
                    calls.append({
                        'target': f"{class_name}.{method}",
                        'method': method,
                        'type': 'local_call'
                    })
        
        return calls
    
    def extract_class_dependencies(self, file_path, class_name):
        """Extract all classes this class depends on"""
        content = self.read_file(file_path)
        dependencies = set()
        
        for match in re.finditer(r'import\s+[\w.]+\.(\w+);', content):
            dependencies.add(match.group(1))
        
        for match in re.finditer(r'(?:private|protected|public)\s+(\w+(?:<[^>]+>)?)\s+\w+;', content):
            type_name = re.sub(r'<[^>]+>', '', match.group(1))
            if type_name != class_name:
                dependencies.add(type_name)
        
        for match in re.finditer(r'@Autowired[^;]*?(\w+)\s+\w+;', content, re.DOTALL):
            dependencies.add(match.group(1))
        
        return dependencies
    
    def extract_external_api_calls(self, file_path, class_name):
        """Extract ALL external API/HTTP calls"""
        content = self.read_file(file_path)
        api_calls = []
        
        rest_patterns = [
            (r'restTemplate\.get(?:ForObject|ForEntity)\s*\(\s*["\']([^"\']+)["\']', 'RestTemplate', 'GET'),
            (r'restTemplate\.post(?:ForObject|ForEntity)\s*\(\s*["\']([^"\']+)["\']', 'RestTemplate', 'POST'),
            (r'HttpGet\s*\(\s*["\']([^"\']+)["\']', 'HttpClient', 'GET'),
            (r'HttpPost\s*\(\s*["\']([^"\']+)["\']', 'HttpClient', 'POST'),
            (r'new\s+URL\s*\(\s*["\']([^"\']+)["\']', 'URLConnection', 'GET'),
        ]
        
        for pattern, client_type, method in rest_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE | re.DOTALL):
                api_calls.append({
                    'class': class_name,
                    'file': Path(file_path).name,
                    'client_type': client_type,
                    'url': match.group(1),
                    'method': method
                })
        
        return api_calls
    
    def extract_spring_component_type(self, file_path, class_name):
        """Extract Spring component type"""
        content = self.read_file(file_path)
        
        spring_annotations = {
            '@Service': 'service',
            '@Controller': 'controller',
            '@RestController': 'rest_controller',
            '@Repository': 'repository',
            '@Component': 'component',
            '@Configuration': 'configuration'
        }
        
        for annotation, comp_type in spring_annotations.items():
            if annotation in content:
                return comp_type
        
        return 'unknown'
    
    def extract_database_operations(self, file_path, class_name):
        """Extract ALL database operations"""
        content = self.read_file(file_path)
        db_ops = []
        
        sql_patterns = [
            (r'(SELECT\s+.+?FROM\s+.+?)(?:;|"|\n)', 'SELECT'),
            (r'(INSERT\s+INTO\s+.+?)(?:;|"|\n)', 'INSERT'),
            (r'(UPDATE\s+.+?SET\s+.+?)(?:;|"|\n)', 'UPDATE'),
            (r'(DELETE\s+FROM\s+.+?)(?:;|"|\n)', 'DELETE'),
        ]
        
        for pattern, op_type in sql_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE | re.DOTALL):
                query = match.group(1).replace('\n', ' ').strip()[:200]
                db_ops.append({
                    'class': class_name,
                    'type': op_type,
                    'query': query,
                    'framework': 'SQL'
                })
        
        jpa_methods = ['save', 'update', 'delete', 'find', 'persist']
        for method in jpa_methods:
            if f'.{method}(' in content:
                db_ops.append({
                    'class': class_name,
                    'type': method.upper(),
                    'framework': 'JPA/Hibernate'
                })
        
        return db_ops
    
    def extract_scheduled_tasks(self, file_path, class_name, method_name):
        """Extract scheduling info"""
        content = self.read_file(file_path)
        tasks = []
        
        scheduled_pattern = rf'@Scheduled\s*\(([^)]+)\)[^{{]*{method_name}\s*\('
        match = re.search(scheduled_pattern, content, re.DOTALL)
        
        if match:
            config = match.group(1)
            task_info = {
                'class': class_name,
                'method': method_name,
                'type': 'scheduled',
                'config': {}
            }
            
            cron = re.search(r'cron\s*=\s*"([^"]+)"', config)
            if cron:
                task_info['config']['cron'] = cron.group(1)
            
            tasks.append(task_info)
        
        if re.search(rf'@Async[^{{]*{method_name}\s*\(', content):
            tasks.append({
                'class': class_name,
                'method': method_name,
                'type': 'async'
            })
        
        return tasks
    
    def extract_error_handling(self, file_path, class_name, method_name):
        """Extract error handling patterns"""
        content = self.read_file(file_path)
        error_info = []
        
        lines = content.split('\n')
        method_pattern = rf'[\w<>\[\]]+\s+{method_name}\s*\('
        method_start = -1
        
        for i, line in enumerate(lines):
            if re.search(method_pattern, line):
                method_start = i
                break
        
        if method_start == -1:
            return error_info
        
        method_body = '\n'.join(lines[method_start:min(method_start + 200, len(lines))])
        
        for match in re.finditer(r'catch\s*\(\s*(\w+)\s+\w+\s*\)', method_body):
            exception_type = match.group(1)
            error_info.append({
                'class': class_name,
                'method': method_name,
                'exception_type': exception_type,
                'pattern': 'try_catch'
            })
        
        return error_info
    
    def extract_config_usage(self, file_path, class_name):
        """Extract configuration property usage"""
        content = self.read_file(file_path)
        configs = []
        
        for match in re.finditer(r'@Value\s*\(\s*"\$\{([^}]+)\}"', content):
            configs.append({
                'class': class_name,
                'property': match.group(1)
            })
        
        return configs

def main(repo_path=None, output_file=None):
    if repo_path is None:
        repo_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REPO_PATH
    if output_file is None:
        output_file = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT_FILE
    
    console.print(Panel(
        "[bold cyan]COMPLETE Deep Code Analysis + SOURCE CODE EXTRACTION[/bold cyan]\n\n"
        f"[yellow]Repository:[/yellow] {repo_path}\n"
        f"[yellow]Output:[/yellow] {output_file}",
        style="blue"
    ))
    
    console.print("\n[bold green]Step 1: Parsing Code Structure...[/bold green]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Parsing files...", total=None)
        parsed_files = parse_repository(repo_path)
        progress.update(task, completed=True)
    
    console.print(f"[green]✓ Parsed {len(parsed_files)} files[/green]\n")
    
    console.print("[bold green]Step 2: Complete Analysis + Source Extraction...[/bold green]\n")
    
    analyzer = CompleteAnalyzer()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Analyzing...", total=len(parsed_files))
        
        for pf in parsed_files:
            file_path = pf.file_path
            
            for cls in pf.classes:
                class_name = cls.name
                analyzer.class_to_file[class_name] = file_path
                
                # EXTRACT CLASS SOURCE CODE
                class_source = analyzer.extract_class_source_code(file_path, class_name)
                analyzer.class_source_code[class_name] = class_source
                
                for method in cls.methods:
                    method_name = method.name
                    full_name = f"{class_name}.{method_name}"
                    
                    # EXTRACT METHOD SOURCE CODE
                    method_source = analyzer.extract_method_source_code(
                        file_path, method_name, method.start_line, method.end_line
                    )
                    analyzer.method_source_code[full_name] = method_source
                    
                    # Extract calls
                    calls = analyzer.extract_full_method_calls(
                        file_path, class_name, method_name,
                        method.start_line, method.end_line
                    )
                    
                    if full_name not in analyzer.method_calls:
                        analyzer.method_calls[full_name] = {'calls': [], 'called_by': []}
                    
                    analyzer.method_calls[full_name]['calls'] = calls
                    
                    for call in calls:
                        target = call.get('target', '')
                        if target:
                            if target not in analyzer.method_calls:
                                analyzer.method_calls[target] = {'calls': [], 'called_by': []}
                            analyzer.method_calls[target]['called_by'].append(full_name)
                    
                    task_list = analyzer.extract_scheduled_tasks(file_path, class_name, method_name)
                    if task_list:
                        analyzer.scheduled_tasks.extend(task_list)
                    
                    error_info = analyzer.extract_error_handling(file_path, class_name, method_name)
                    if error_info:
                        analyzer.error_handling.extend(error_info)
                
                deps = analyzer.extract_class_dependencies(file_path, class_name)
                analyzer.class_dependencies[class_name] = deps
                
                api_calls = analyzer.extract_external_api_calls(file_path, class_name)
                analyzer.external_apis.extend(api_calls)
                
                db_ops = analyzer.extract_database_operations(file_path, class_name)
                analyzer.database_operations.extend(db_ops)
                
                comp_type = analyzer.extract_spring_component_type(file_path, class_name)
                if comp_type:
                    analyzer.spring_components[class_name] = comp_type
                
                configs = analyzer.extract_config_usage(file_path, class_name)
                analyzer.config_usage.extend(configs)
            
            progress.update(task, advance=1)
    
    # Build comprehensive report
    complete_analysis = {
        "summary": {
            "total_methods": len(analyzer.method_calls),
            "total_classes": len(analyzer.class_source_code),
            "total_method_calls": sum(len(m['calls']) for m in analyzer.method_calls.values()),
            "external_api_calls": len(analyzer.external_apis),
            "database_operations": len(analyzer.database_operations),
            "scheduled_tasks": len(analyzer.scheduled_tasks),
            "methods_with_source": len(analyzer.method_source_code),
            "classes_with_source": len(analyzer.class_source_code)
        },
        
        # CORE DATA WITH SOURCE CODE
        "methods": {
            method_name: {
                "calls": data['calls'],
                "called_by": data['called_by'],
                "source": analyzer.method_source_code.get(method_name, {})
            }
            for method_name, data in analyzer.method_calls.items()
        },
        
        "classes": {
            class_name: {
                "dependencies": list(analyzer.class_dependencies.get(class_name, [])),
                "component_type": analyzer.spring_components.get(class_name, 'unknown'),
                "source": source_data
            }
            for class_name, source_data in analyzer.class_source_code.items()
        },
        
        "external_api_calls": analyzer.external_apis,
        "database_operations": analyzer.database_operations,
        "scheduled_tasks": analyzer.scheduled_tasks,
        "error_handling": analyzer.error_handling,
        "configuration_usage": analyzer.config_usage,
        
        "class_to_file_mapping": {k: str(v) for k, v in analyzer.class_to_file.items()}
    }
    
    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(complete_analysis, f, indent=2)
    
    console.print(f"\n[green]✓ Analysis saved to:[/green] {output_file}\n")
    
    # Display summary
    console.print(Panel(
        f"[bold green]✓ COMPLETE Analysis with SOURCE CODE![/bold green]\n\n"
        f"[cyan]Methods Analyzed:[/cyan] {len(analyzer.method_calls)}\n"
        f"[cyan]Classes Analyzed:[/cyan] {len(analyzer.class_source_code)}\n"
        f"[cyan]Methods with Source Code:[/cyan] {len(analyzer.method_source_code)}\n"
        f"[cyan]Classes with Source Code:[/cyan] {len(analyzer.class_source_code)}\n"
        f"[cyan]Method Calls Traced:[/cyan] {sum(len(m['calls']) for m in analyzer.method_calls.values())}\n"
        f"[cyan]External API Calls:[/cyan] {len(analyzer.external_apis)}\n"
        f"[cyan]Database Operations:[/cyan] {len(analyzer.database_operations)}\n"
        f"[cyan]Scheduled Tasks:[/cyan] {len(analyzer.scheduled_tasks)}\n\n"
        f"[yellow]✨ Now includes full source code for all methods and classes![/yellow]",
        style="green",
        title="Analysis Complete"
    ))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Analysis interrupted[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Error: {e}[/bold red]")
        import traceback
        traceback.print_exc()