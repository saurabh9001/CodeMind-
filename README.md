# Code Parser System

Simple multi-language code parser using Tree-sitter.

## Structure
```
p/
├── analyze_sample.py    # Main script
├── config.py           # Configuration
├── parser/             # Parser module
│   ├── code_parser.py
│   ├── language_parsers.py
│   └── models.py
├── sample_code/        # Test code samples
├── requirements.txt    # Dependencies
└── venv/              # Virtual environment
```

## Quick Start

```bash
# Activate environment
source venv/bin/activate

# Run parser
python analyze_sample.py
```

## What it does
- Parses Java, Python, TypeScript code
- Extracts classes, functions, methods
- Shows code structure in tables
- Displays parsing statistics

## Output
- Files parsed
- Classes extracted
- Functions/methods found
- Imports discovered
