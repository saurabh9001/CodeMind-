"""
COMPLETE Deep Code Analysis - All Relationships & Behaviors
Extracts EVERYTHING: Full call graphs, all dependencies, all patterns
Self-contained version with inline parsing

Usage:
  python ultimate_code_analyzer.py [repo_path] [output_file]
  
Examples:
  python ultimate_code_analyzer.py
  python ultimate_code_analyzer.py /path/to/your/java/code
  python ultimate_code_analyzer.py /path/to/java/code output.json
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

# ============================================================================
# CONFIGURATION - Change these paths as needed
# ============================================================================
DEFAULT_REPO_PATH = "/Users/home/Desktop/CODEMIND/project"
DEFAULT_OUTPUT_FILE = "/Users/home/Desktop/CODEMIND/platform/1.parser/output parser/f.json"
# ============================================================================

# Simple parsed file structure
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
                # Extract class name
                class_name = None
                for child in node.children:
                    if child.type == 'identifier':
                        class_name = child.text.decode('utf8')
                        break
                
                if class_name:
                    parsed_class = ParsedClass(class_name)
                    
                    # Find methods
                    body = None
                    for child in node.children:
                        if child.type == 'class_body':
                            body = child
                            break
                    
                    if body:
                        for child in body.children:
                            if child.type == 'method_declaration':
                                # Extract method name
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
        # Fallback: return empty parsed file
        return ParsedFile(file_path)

def parse_repository(repo_path):
    """Parse all Java files in repository"""
    parsed_files = []
    repo_path = Path(repo_path)
    
    for java_file in repo_path.rglob('*.java'):
        if '/test/' not in str(java_file):  # Skip test files
            parsed = parse_java_file(java_file)
            if parsed.classes:  # Only include files with classes
                parsed_files.append(parsed)
    
    return parsed_files

class CompleteAnalyzer:
    def __init__(self):
        # 1. Full method call graph
        self.method_calls = {}  # method_full_name -> {calls: [], called_by: []}
        
        # 2. Cross-class dependencies
        self.class_dependencies = defaultdict(set)  # class -> set of classes it depends on
        
        # 3. External API calls
        self.external_apis = []  # List of all HTTP/REST calls
        
        # 4. Database interactions
        self.database_operations = []  # List of all DB operations
        
        # 5. Scheduler/Async tasks
        self.scheduled_tasks = []  # All scheduled/async methods
        
        # 6. Spring component classification
        self.spring_components = {}  # class -> type (Service/Repository/etc)
        
        # 7. Domain model relationships
        self.model_usage = defaultdict(list)  # model_class -> [classes that use it]
        
        # 8. Error handling
        self.error_handling = []  # All try-catch blocks and retry logic
        
        # 9. Configuration usage
        self.config_usage = []  # Where configs are loaded
        
        # 10. Event routing
        self.event_routing = []  # Event -> Worker mappings
        
        # 11. OPTIONAL ADVANCED LAYERS
        self.method_complexity = {}  # method -> cyclomatic complexity
        self.database_schema_mapping = defaultdict(dict)  # table -> {dao: [], service: []}
        self.api_endpoints = []  # REST endpoint mappings
        self.domain_clusters = defaultdict(list)  # domain/concept -> [classes]
        
        # Support data structures
        self.class_to_file = {}
        self.method_to_class = {}
        self.file_contents = {}  # Cache file contents
    
    def read_file(self, file_path):
        """Read and cache file content"""
        if file_path not in self.file_contents:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.file_contents[file_path] = f.read()
            except:
                self.file_contents[file_path] = ""
        return self.file_contents[file_path]
    
    def extract_full_method_calls(self, file_path, class_name, method_name, start_line, end_line):
        """Extract ALL method calls from a method body"""
        content = self.read_file(file_path)
        lines = content.split('\n')
        
        # Get method body
        method_body = '\n'.join(lines[start_line-1:end_line])
        
        calls = []
        
        # Pattern 1: object.method()
        for match in re.finditer(r'(\w+)\.(\w+)\s*\(', method_body):
            obj = match.group(1)
            method = match.group(2)
            if method not in ['if', 'for', 'while', 'switch', 'return']:
                calls.append({
                    'target': f"{obj}.{method}",
                    'object': obj,
                    'method': method,
                    'type': 'instance_call'
                })
        
        # Pattern 2: ClassName.staticMethod()
        for match in re.finditer(r'([A-Z]\w+)\.(\w+)\s*\(', method_body):
            cls = match.group(1)
            method = match.group(2)
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
                if not method[0].isupper():  # Not a constructor
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
        
        # From imports
        for match in re.finditer(r'import\s+[\w.]+\.(\w+);', content):
            dependencies.add(match.group(1))
        
        # From field declarations
        for match in re.finditer(r'(?:private|protected|public)\s+(\w+(?:<[^>]+>)?)\s+\w+;', content):
            type_name = match.group(1)
            type_name = re.sub(r'<[^>]+>', '', type_name)  # Remove generics
            if type_name != class_name:
                dependencies.add(type_name)
        
        # From @Autowired
        for match in re.finditer(r'@Autowired[^;]*?(\w+)\s+\w+;', content, re.DOTALL):
            dependencies.add(match.group(1))
        
        return dependencies
    
    def extract_external_api_calls(self, file_path, class_name):
        """Extract ALL external API/HTTP calls - ENHANCED VERSION"""
        content = self.read_file(file_path)
        api_calls = []
        
        # Pattern 1: RestTemplate (Spring's HTTP client)
        rest_patterns = [
            r'restTemplate\.get(?:ForObject|ForEntity)\s*\(\s*["\']([^"\']+)["\']',
            r'restTemplate\.post(?:ForObject|ForEntity|ForLocation)\s*\(\s*["\']([^"\']+)["\']',
            r'restTemplate\.put\s*\(\s*["\']([^"\']+)["\']',
            r'restTemplate\.delete\s*\(\s*["\']([^"\']+)["\']',
            r'restTemplate\.exchange\s*\(\s*["\']([^"\']+)["\']',
        ]
        
        for pattern in rest_patterns:
            for match in re.finditer(pattern, content, re.DOTALL | re.IGNORECASE):
                api_calls.append({
                    'class': class_name,
                    'file': Path(file_path).name,
                    'client_type': 'RestTemplate',
                    'url': match.group(1),
                    'pattern': 'direct_call'
                })
        
        # Pattern 2: HttpClient (Apache HttpComponents)
        http_patterns = [
            r'HttpGet\s*\(\s*["\']([^"\']+)["\']',
            r'HttpPost\s*\(\s*["\']([^"\']+)["\']',
            r'HttpPut\s*\(\s*["\']([^"\']+)["\']',
            r'HttpDelete\s*\(\s*["\']([^"\']+)["\']',
            r'httpClient\.execute\s*\(',
        ]
        
        for pattern in http_patterns:
            for match in re.finditer(pattern, content, re.DOTALL | re.IGNORECASE):
                url = match.group(1) if match.groups() else 'dynamic'
                api_calls.append({
                    'class': class_name,
                    'file': Path(file_path).name,
                    'client_type': 'HttpClient',
                    'url': url,
                    'pattern': 'apache_http'
                })
        
        # Pattern 3: URL/URLConnection (Java built-in)
        url_patterns = [
            r'new\s+URL\s*\(\s*["\']([^"\']+)["\']',
            r'URL\.openConnection\s*\(',
            r'HttpURLConnection',
        ]
        
        for pattern in url_patterns:
            for match in re.finditer(pattern, content, re.DOTALL | re.IGNORECASE):
                url = match.group(1) if match.groups() else 'connection'
                api_calls.append({
                    'class': class_name,
                    'file': Path(file_path).name,
                    'client_type': 'URLConnection',
                    'url': url,
                    'pattern': 'java_net'
                })
        
        # Pattern 4: Feed URLs and properties (OpenELIS specific)
        feed_patterns = [
            r'(?:feed\.uri|feedUri|feed\.url|feedUrl)\s*=\s*["\']([^"\']+)["\']',
            r'@Value\s*\(\s*"\$\{([^}]*feed[^}]*)\}"',
            r'getProperty\s*\(\s*["\']([^"\']*feed[^"\']*)["\']',
        ]
        
        for pattern in feed_patterns:
            for match in re.finditer(pattern, content, re.DOTALL | re.IGNORECASE):
                api_calls.append({
                    'class': class_name,
                    'file': Path(file_path).name,
                    'client_type': 'FeedClient',
                    'url': match.group(1),
                    'pattern': 'feed_url_config'
                })
        
        # Pattern 5: AtomFeed / EventLog patterns
        atom_patterns = [
            r'AtomFeed',
            r'EventLog',
            r'FeedClient',
            r'FailedEvent',
        ]
        
        for pattern in atom_patterns:
            if re.search(pattern, content):
                api_calls.append({
                    'class': class_name,
                    'file': Path(file_path).name,
                    'client_type': 'AtomFeedClient',
                    'url': 'event_feed',
                    'pattern': 'atom_feed'
                })
                break
        
        return api_calls
    
    def extract_spring_component_type(self, file_path, class_name):
        """Extract Spring component type (Service, Controller, Repository, etc.)"""
        content = self.read_file(file_path)
        
        # Check for Spring annotations
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
        
        # SQL queries
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
                    'file': Path(file_path).name,
                    'type': op_type,
                    'query': query,
                    'framework': 'SQL'
                })
        
        # Hibernate/JPA operations
        jpa_methods = ['save', 'update', 'delete', 'find', 'persist', 'merge', 'remove', 'createQuery', 'createNativeQuery']
        for method in jpa_methods:
            if f'.{method}(' in content:
                db_ops.append({
                    'class': class_name,
                    'file': Path(file_path).name,
                    'type': method.upper(),
                    'framework': 'JPA/Hibernate',
                    'method': method
                })
        
        return db_ops
    
    def extract_scheduled_tasks(self, file_path, class_name, method_name):
        """Extract scheduling and async information - ENHANCED VERSION"""
        content = self.read_file(file_path)
        tasks = []
        
        # Pattern 1: @Scheduled annotation (Spring scheduled tasks)
        scheduled_pattern = rf'@Scheduled\s*\(([^)]+)\)[^{{]*?(?:public|private|protected)?\s+[\w<>\[\]]+\s+{method_name}\s*\('
        match = re.search(scheduled_pattern, content, re.DOTALL)
        
        if match:
            config = match.group(1)
            
            task_info = {
                'class': class_name,
                'method': method_name,
                'file': Path(file_path).name,
                'type': 'scheduled',
                'config': {}
            }
            
            # Extract cron
            cron = re.search(r'cron\s*=\s*"([^"]+)"', config)
            if cron:
                task_info['config']['cron'] = cron.group(1)
            
            # Extract fixedDelay
            delay = re.search(r'fixedDelay\s*=\s*(\d+)', config)
            if delay:
                task_info['config']['fixedDelay'] = delay.group(1)
            
            # Extract fixedRate
            rate = re.search(r'fixedRate\s*=\s*(\d+)', config)
            if rate:
                task_info['config']['fixedRate'] = rate.group(1)
            
            # Extract initialDelay
            initial = re.search(r'initialDelay\s*=\s*(\d+)', config)
            if initial:
                task_info['config']['initialDelay'] = initial.group(1)
            
            tasks.append(task_info)
        
        # Pattern 2: @Async annotation (Spring async methods)
        if re.search(rf'@Async[^{{]*?{method_name}\s*\(', content):
            tasks.append({
                'class': class_name,
                'method': method_name,
                'file': Path(file_path).name,
                'type': 'async',
                'config': {'executor': 'default'}
            })
        
        # Pattern 3: EventListener / ApplicationListener (event-driven)
        event_patterns = [
            r'@EventListener',
            r'implements\s+ApplicationListener',
            r'onApplicationEvent',
        ]
        
        for pattern in event_patterns:
            if re.search(pattern, content):
                tasks.append({
                    'class': class_name,
                    'method': method_name,
                    'file': Path(file_path).name,
                    'type': 'event_listener',
                    'config': {'trigger': 'application_event'}
                })
                break
        
        # Pattern 4: Worker classes (Feed workers, Event workers)
        worker_patterns = [
            r'Worker',
            r'FeedTask',
            r'EventWorker',
            r'JobExecutor',
        ]
        
        for pattern in worker_patterns:
            if re.search(pattern, class_name):
                tasks.append({
                    'class': class_name,
                    'method': method_name,
                    'file': Path(file_path).name,
                    'type': 'worker',
                    'config': {'worker_type': pattern}
                })
                break
        
        # Pattern 5: Quartz/Scheduled jobs
        if 'Job' in class_name or 'Task' in class_name:
            if re.search(r'execute|run|process', method_name, re.IGNORECASE):
                tasks.append({
                    'class': class_name,
                    'method': method_name,
                    'file': Path(file_path).name,
                    'type': 'job',
                    'config': {'job_type': 'scheduled_job'}
                })
        
        return tasks
    
    def extract_error_handling(self, file_path, class_name, method_name):
        """Extract error handling patterns - ENHANCED VERSION"""
        content = self.read_file(file_path)
        error_info = []
        
        # Get method body
        lines = content.split('\n')
        method_pattern = rf'(?:public|private|protected)?\s+[\w<>\[\]]+\s+{method_name}\s*\([^)]*\)'
        method_start = -1
        
        for i, line in enumerate(lines):
            if re.search(method_pattern, line):
                method_start = i
                break
        
        if method_start == -1:
            return error_info
        
        # Extract method body (simplified - until next method or class end)
        method_body = '\n'.join(lines[method_start:min(method_start + 200, len(lines))])
        
        # Pattern 1: try-catch blocks
        catch_patterns = [
            r'catch\s*\(\s*(\w+(?:Exception|Error|Throwable)?)\s+(\w+)\s*\)',
            r'catch\s*\(\s*(\w+)\s+(\w+)\s*\)',
        ]
        
        for pattern in catch_patterns:
            for match in re.finditer(pattern, method_body):
                exception_type = match.group(1)
                exception_var = match.group(2)
                
                # Check what happens in catch block
                catch_block_start = match.end()
                catch_snippet = method_body[catch_block_start:catch_block_start + 300]
                
                handles_retry = bool(re.search(r'retry|Retry|FailedEvent', catch_snippet))
                logs_error = bool(re.search(r'log\.|logger\.|LOG\.', catch_snippet, re.IGNORECASE))
                rethrows = bool(re.search(r'throw\s+', catch_snippet))
                
                error_info.append({
                    'class': class_name,
                    'method': method_name,
                    'file': Path(file_path).name,
                    'exception_type': exception_type,
                    'handles_retry': handles_retry,
                    'logs_error': logs_error,
                    'rethrows': rethrows,
                    'pattern': 'try_catch'
                })
        
        # Pattern 2: @Retryable annotation
        if re.search(r'@Retryable', content):
            error_info.append({
                'class': class_name,
                'method': method_name,
                'file': Path(file_path).name,
                'exception_type': 'RetryableOperation',
                'handles_retry': True,
                'pattern': 'annotation_retry'
            })
        
        # Pattern 3: FailedEvent / Retry classes
        retry_keywords = [
            'FailedEvent',
            'RetryPolicy',
            'RetryTemplate',
            'FailedEventsFeedClient',
            'retry',
        ]
        
        for keyword in retry_keywords:
            if keyword in method_body:
                error_info.append({
                    'class': class_name,
                    'method': method_name,
                    'file': Path(file_path).name,
                    'exception_type': 'FailureHandling',
                    'handles_retry': True,
                    'pattern': f'retry_mechanism_{keyword}'
                })
                break
        
        # Pattern 4: Circuit breaker patterns
        if re.search(r'@CircuitBreaker|@Fallback|@HystrixCommand', content):
            error_info.append({
                'class': class_name,
                'method': method_name,
                'file': Path(file_path).name,
                'exception_type': 'CircuitBreaker',
                'handles_retry': True,
                'pattern': 'circuit_breaker'
            })
        
        return error_info
        return None
    
    def extract_model_usage(self, file_path, class_name):
        """Extract which domain models this class uses"""
        content = self.read_file(file_path)
        models = set()
        
        # Look for field declarations, parameters, return types
        # Common model patterns
        model_keywords = ['Patient', 'Visit', 'Encounter', 'Observation', 'Order', 'Concept', 'Person', 'Provider', 'Note', 'Document']
        
        for keyword in model_keywords:
            if keyword in content:
                models.add(keyword)
        
        return list(models)
    
    def extract_config_usage(self, file_path, class_name):
        """Extract configuration property usage"""
        content = self.read_file(file_path)
        configs = []
        
        # @Value annotations
        for match in re.finditer(r'@Value\s*\(\s*"\$\{([^}]+)\}"', content):
            property_name = match.group(1)
            configs.append({
                'class': class_name,
                'file': Path(file_path).name,
                'type': 'property',
                'property': property_name
            })
        
        # Properties file reading
        if 'Properties' in content or 'getProperty' in content:
            configs.append({
                'class': class_name,
                'file': Path(file_path).name,
                'type': 'properties_file',
                'reads_config': True
            })
        
        return configs
    
    def calculate_cyclomatic_complexity(self, parsed_files):
        """OPTIONAL: Calculate cyclomatic complexity for each method (identifies risky code)"""
        for pf in parsed_files:
            content = self.read_file(pf.file_path)
            
            for cls in pf.classes:
                for method in cls.methods:
                    full_name = f"{cls.name}.{method.name}"
                    lines = content.split('\n')
                    method_body = '\n'.join(lines[method.start_line-1:method.end_line])
                    
                    complexity = 1  # Base complexity
                    decision_keywords = [r'\bif\b', r'\belse\s+if\b', r'\bfor\b', r'\bwhile\b', 
                                       r'\bcase\b', r'\bcatch\b', r'\&\&', r'\|\|', r'\?']
                    
                    for keyword in decision_keywords:
                        complexity += len(re.findall(keyword, method_body))
                    
                    risk = "low" if complexity <= 5 else "medium" if complexity <= 10 else "high" if complexity <= 20 else "very_high"
                    
                    self.method_complexity[full_name] = {
                        'complexity': complexity,
                        'risk': risk,
                        'class': cls.name,
                        'method': method.name
                    }
    
    def map_database_schema(self, parsed_files):
        """OPTIONAL: Map database tables to DAOs and Services"""
        for pf in parsed_files:
            content = self.read_file(pf.file_path)
            
            for cls in pf.classes:
                class_name = cls.name
                comp_type = self.spring_components.get(class_name, '')
                
                table_patterns = [r'FROM\s+(\w+)', r'JOIN\s+(\w+)', r'INTO\s+(\w+)', 
                                r'@Table\s*\(\s*name\s*=\s*"([^"]+)"']
                
                tables_found = set()
                for pattern in table_patterns:
                    for match in re.finditer(pattern, content, re.IGNORECASE):
                        tables_found.add(match.group(1))
                
                for table in tables_found:
                    if 'dao' not in self.database_schema_mapping[table]:
                        self.database_schema_mapping[table]['dao'] = []
                    if 'service' not in self.database_schema_mapping[table]:
                        self.database_schema_mapping[table]['service'] = []
                    
                    if comp_type == 'Repository' or 'Dao' in class_name:
                        self.database_schema_mapping[table]['dao'].append(class_name)
                    elif comp_type == 'Service':
                        self.database_schema_mapping[table]['service'].append(class_name)
    
    def map_api_endpoints(self, parsed_files):
        """OPTIONAL: Map REST API endpoints to controllers"""
        for pf in parsed_files:
            content = self.read_file(pf.file_path)
            
            for cls in pf.classes:
                class_name = cls.name
                comp_type = self.spring_components.get(class_name, '')
                
                if comp_type not in ['Controller', 'RestController']:
                    continue
                
                base_path = ""
                class_mapping = re.search(r'@RequestMapping\s*\(\s*["\']([^"\']+)["\']', content)
                if class_mapping:
                    base_path = class_mapping.group(1)
                
                for method in cls.methods:
                    method_name = method.name
                    lines = content.split('\n')
                    method_area = '\n'.join(lines[max(0, method.start_line-10):method.start_line+5])
                    
                    rest_patterns = [
                        (r'@GetMapping\s*\(\s*["\']([^"\']+)["\']', 'GET'),
                        (r'@PostMapping\s*\(\s*["\']([^"\']+)["\']', 'POST'),
                        (r'@PutMapping\s*\(\s*["\']([^"\']+)["\']', 'PUT'),
                        (r'@DeleteMapping\s*\(\s*["\']([^"\']+)["\']', 'DELETE'),
                    ]
                    
                    for pattern, http_method in rest_patterns:
                        for match in re.finditer(pattern, method_area):
                            path = match.group(1)
                            self.api_endpoints.append({
                                'path': base_path + path,
                                'method': http_method,
                                'controller': class_name,
                                'handler': method_name
                            })
    
    def detect_domain_clusters(self, parsed_files):
        """OPTIONAL: Detect bounded contexts and domain clustering"""
        domain_keywords = {
            'Patient Management': ['Patient', 'Visit', 'Admission', 'Discharge'],
            'Clinical': ['Encounter', 'Diagnosis', 'Observation', 'Vital'],
            'Medication': ['Drug', 'Order', 'Prescription', 'Medication'],
            'Laboratory': ['Lab', 'Test', 'Result', 'Specimen'],
            'Billing': ['Bill', 'Invoice', 'Payment', 'Charge'],
            'Document Management': ['Document', 'Report', 'Note', 'Attachment'],
            'User Management': ['User', 'Provider', 'Staff', 'Role'],
        }
        
        for pf in parsed_files:
            for cls in pf.classes:
                class_name = cls.name
                
                for domain, keywords in domain_keywords.items():
                    for keyword in keywords:
                        if keyword.lower() in class_name.lower():
                            self.domain_clusters[domain].append({
                                'class': class_name,
                                'type': self.spring_components.get(class_name, 'Other')
                            })
                            break

def main(repo_path=None, output_file=None):
    # Use command-line args or defaults
    if repo_path is None:
        repo_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REPO_PATH
    if output_file is None:
        output_file = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT_FILE
    
    console.print(Panel(
        "[bold cyan]COMPLETE Deep Code Analysis[/bold cyan]\n"
        "Extracting: ALL relationships, ALL behaviors, ALL patterns\n\n"
        f"[yellow]Repository:[/yellow] {repo_path}\n"
        f"[yellow]Output:[/yellow] {output_file}",
        style="blue"
    ))
    
    # Parse repository
    console.print("\n[bold green]Step 1: Parsing Code Structure...[/bold green]")
    sample_path = repo_path
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task("Parsing files...", total=None)
        parsed_files = parse_repository(sample_path)
        progress.update(task, completed=True)
    
    console.print(f"[green]✓ Parsed {len(parsed_files)} files[/green]\n")
    
    # Complete analysis
    console.print("[bold green]Step 2: Complete Deep Analysis...[/bold green]\n")
    
    analyzer = CompleteAnalyzer()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task("Analyzing all patterns...", total=len(parsed_files))
        
        for pf in parsed_files:
            file_path = pf.file_path
            
            for cls in pf.classes:
                class_name = cls.name
                analyzer.class_to_file[class_name] = file_path
                
                # 1. Full method call graph
                for method in cls.methods:
                    method_name = method.name
                    full_name = f"{class_name}.{method_name}"
                    analyzer.method_to_class[method_name] = class_name
                    
                    calls = analyzer.extract_full_method_calls(
                        file_path, class_name, method_name,
                        method.start_line, method.end_line
                    )
                    
                    if full_name not in analyzer.method_calls:
                        analyzer.method_calls[full_name] = {'calls': [], 'called_by': []}
                    
                    analyzer.method_calls[full_name]['calls'] = calls
                    
                    # Build reverse call graph
                    for call in calls:
                        target = call.get('target', '')
                        if target:
                            if target not in analyzer.method_calls:
                                analyzer.method_calls[target] = {'calls': [], 'called_by': []}
                            analyzer.method_calls[target]['called_by'].append(full_name)
                    
                    # 5. Scheduled tasks (returns list now)
                    task_list = analyzer.extract_scheduled_tasks(file_path, class_name, method_name)
                    if task_list:
                        analyzer.scheduled_tasks.extend(task_list)
                    
                    # 8. Error handling (returns list)
                    error_info = analyzer.extract_error_handling(file_path, class_name, method_name)
                    if error_info:
                        analyzer.error_handling.extend(error_info)
                
                # 2. Class dependencies
                deps = analyzer.extract_class_dependencies(file_path, class_name)
                analyzer.class_dependencies[class_name] = deps
                
                # 3. External API calls
                api_calls = analyzer.extract_external_api_calls(file_path, class_name)
                analyzer.external_apis.extend(api_calls)
                
                # 4. Database operations
                db_ops = analyzer.extract_database_operations(file_path, class_name)
                analyzer.database_operations.extend(db_ops)
                
                # 6. Spring component classification
                comp_type = analyzer.extract_spring_component_type(file_path, class_name)
                if comp_type:
                    analyzer.spring_components[class_name] = comp_type
                
                # 7. Domain model usage
                models = analyzer.extract_model_usage(file_path, class_name)
                for model in models:
                    analyzer.model_usage[model].append(class_name)
                
                # 9. Configuration usage
                configs = analyzer.extract_config_usage(file_path, class_name)
                analyzer.config_usage.extend(configs)
            
            progress.update(task, advance=1)
    
    # OPTIONAL ADVANCED ANALYSIS
    console.print("\n[bold cyan]Step 3: Optional Advanced Analysis...[/bold cyan]")
    analyzer.calculate_cyclomatic_complexity(parsed_files)
    analyzer.map_database_schema(parsed_files)
    analyzer.map_api_endpoints(parsed_files)
    analyzer.detect_domain_clusters(parsed_files)
    console.print("[green]✓ Advanced analysis complete[/green]\n")
    
    # Build comprehensive report
    complete_analysis = {
        "summary": {
            "total_methods_analyzed": len(analyzer.method_calls),
            "total_method_calls": sum(len(m['calls']) for m in analyzer.method_calls.values()),
            "total_classes": len(analyzer.class_to_file),
            "classes_with_dependencies": len(analyzer.class_dependencies),
            "external_api_calls": len(analyzer.external_apis),
            "database_operations": len(analyzer.database_operations),
            "scheduled_tasks": len(analyzer.scheduled_tasks),
            "spring_components": len(analyzer.spring_components),
            "error_handlers": len(analyzer.error_handling),
            "config_users": len(analyzer.config_usage),
            "method_complexity_analyzed": len(analyzer.method_complexity),
            "high_risk_methods": len([m for m, c in analyzer.method_complexity.items() if c['risk'] in ['high', 'very_high']]),
            "database_tables_mapped": len(analyzer.database_schema_mapping),
            "api_endpoints_mapped": len(analyzer.api_endpoints),
            "domain_clusters": len(analyzer.domain_clusters)
        },
        
        # 1. Full method call graph
        "method_call_graph": analyzer.method_calls,
        
        # 2. Cross-class dependencies
        "class_dependencies": {k: list(v) for k, v in analyzer.class_dependencies.items()},
        
        # 3. External API calls
        "external_api_calls": analyzer.external_apis,
        
        # 4. Database operations
        "database_operations": analyzer.database_operations,
        
        # 5. Scheduled tasks
        "scheduled_tasks": analyzer.scheduled_tasks,
        
        # 6. Spring components
        "spring_components": analyzer.spring_components,
        
        # 7. Domain model usage
        "domain_model_usage": dict(analyzer.model_usage),
        
        # 8. Error handling
        "error_handling": analyzer.error_handling,
        
        # 9. Configuration usage
        "configuration_usage": analyzer.config_usage,
        
        # OPTIONAL ADVANCED LAYERS
        "method_complexity": analyzer.method_complexity,
        "database_schema_mapping": {k: v for k, v in analyzer.database_schema_mapping.items()},
        "api_endpoints": analyzer.api_endpoints,
        "domain_clusters": {k: v for k, v in analyzer.domain_clusters.items()},
        
        # Support mappings
        "class_to_file_mapping": {k: str(v) for k, v in analyzer.class_to_file.items()}
    }
    
    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(complete_analysis, f, indent=2)
    
    console.print(f"\n[green]✓ Complete analysis saved to:[/green] {output_file}\n")
    
    # Display summary
    console.print(Panel(
        f"[bold green]✓ COMPLETE Deep Analysis Done![/bold green]\n\n"
        f"[cyan]1️⃣ Full Method Call Graph:[/cyan]\n"
        f"  • {len(analyzer.method_calls)} methods analyzed\n"
        f"  • {sum(len(m['calls']) for m in analyzer.method_calls.values())} method calls traced\n"
        f"  • Bidirectional: calls + called_by\n\n"
        f"[cyan]2️⃣ Cross-Class Dependencies:[/cyan]\n"
        f"  • {len(analyzer.class_dependencies)} classes with dependencies\n"
        f"  • Complete dependency graph\n\n"
        f"[cyan]3️⃣ External API Calls:[/cyan]\n"
        f"  • {len(analyzer.external_apis)} external API invocations\n"
        f"  • Includes URLs and client types\n\n"
        f"[cyan]4️⃣ Database Operations:[/cyan]\n"
        f"  • {len(analyzer.database_operations)} database operations\n"
        f"  • SQL queries + JPA/Hibernate calls\n\n"
        f"[cyan]5️⃣ Scheduled Tasks:[/cyan]\n"
        f"  • {len(analyzer.scheduled_tasks)} scheduled/async workers\n"
        f"  • Includes cron expressions\n\n"
        f"[cyan]6️⃣ Spring Components:[/cyan]\n"
        f"  • {len(analyzer.spring_components)} classified components\n"
        f"  • Service/Repository/Controller layers\n\n"
        f"[cyan]7️⃣ Domain Model Usage:[/cyan]\n"
        f"  • {len(analyzer.model_usage)} domain models tracked\n"
        f"  • Which classes use which models\n\n"
        f"[cyan]8️⃣ Error Handling:[/cyan]\n"
        f"  • {len(analyzer.error_handling)} error handlers\n"
        f"  • Try-catch + retry patterns\n\n"
        f"[cyan]9️⃣ Configuration Usage:[/cyan]\n"
        f"  • {len(analyzer.config_usage)} config usages\n"
        f"  • Property files + @Value\n\n"
        f"[cyan]🔟 OPTIONAL ADVANCED LAYERS:[/cyan]\n"
        f"  • {len(analyzer.method_complexity)} methods with complexity analysis\n"
        f"  • {len([m for m, c in analyzer.method_complexity.items() if c['risk'] in ['high', 'very_high']])} high-risk methods\n"
        f"  • {len(analyzer.database_schema_mapping)} database tables mapped\n"
        f"  • {len(analyzer.api_endpoints)} REST API endpoints\n"
        f"  • {len(analyzer.domain_clusters)} domain clusters\n\n"  ,
        style="green",
        title="Complete Analysis"
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
