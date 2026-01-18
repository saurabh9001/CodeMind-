# CodeMind — Intelligent Code Analysis & RAG Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**CodeMind** is an advanced code intelligence platform that transforms any codebase into a searchable, LLM-powered knowledge base. By combining static code analysis, semantic chunking, vector embeddings, and retrieval-augmented generation (RAG), CodeMind enables deep understanding of complex codebases through natural language queries.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Demo Video](#demo-video)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Pipeline Components](#pipeline-components)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

CodeMind converts your codebase into a searchable, intelligent knowledge base by:
- **Parsing** source code to extract structure, methods, classes, and call graphs
- **Chunking** the analysis into semantically-rich segments with comprehensive metadata
- **Embedding** chunks using advanced code-aware models (GraphCodeBERT/OpenAI)
- **Indexing** embeddings in FAISS for efficient similarity search
- **Retrieving** relevant code context and generating answers via LLM-powered QA

This enables developers to:
- 🔍 Query codebases in natural language
- 📊 Understand complex code relationships and dependencies
- 🧠 Get AI-powered insights about code behavior and patterns
- ⚡ Accelerate code review, onboarding, and maintenance

---

## ✨ Key Features

### 🔬 Deep Code Analysis
- **AST-based parsing** using Tree-sitter for accurate code structure extraction
- **Complete call graph** analysis (method → caller/callee relationships)
- **Dependency tracking** at class and method levels
- **Spring Framework** component detection and analysis
- **Database operations** identification and tracking
- **Error handling** pattern recognition
- **Scheduled tasks** and configuration usage detection

### 🧩 Intelligent Chunking
- **Metadata-enriched chunks** with contextual information
- **Semantic descriptions** that preserve code intent
- **Complexity analysis** and risk assessment per method
- **Cross-reference tracking** (callers, callees, dependencies)

### 🚀 Advanced Embeddings
- **Code-aware embeddings** using Microsoft's GraphCodeBERT
- **OpenAI integration** for high-dimensional embeddings (3072 dims)
- **FAISS indexing** for lightning-fast similarity search
- **Batch processing** for efficient large codebase handling

### 💬 Interactive QA System
- **Natural language queries** over your codebase
- **Context-aware retrieval** using vector similarity
- **LLM-powered answers** with source code references
- **Rich metadata** in responses (complexity, dependencies, call graphs)

---

## 🏗️ Architecture

![Architecture Flow Diagram](./Flow%20diagram.png)

### Pipeline Overview

```
┌─────────────────┐
│  Java Source    │
│     Code        │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  1. PARSER (ultimate_code_analyzer.py)             │
│  - Tree-sitter AST parsing                          │
│  - Extract classes, methods, call graphs            │
│  - Analyze Spring components, DB ops, patterns      │
│  Output: f.json (complete analysis)                 │
└────────┬────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  2. CHUNKER (metadata_enriched_chunker.py)         │
│  - Convert analysis to semantic chunks              │
│  - Enrich with metadata (complexity, dependencies)  │
│  - Generate contextual descriptions                 │
│  Output: enriched_chunks.json                       │
└────────┬────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  3. EMBEDDER (codebert_embedder.py)                │
│  - Generate vector embeddings (GraphCodeBERT/OpenAI)│
│  - Build FAISS index for similarity search          │
│  Output: code_index.faiss + metadata.npy            │
└────────┬────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  4. RETRIEVER (query_qa_v2.py)                     │
│  - Embed user queries                               │
│  - Retrieve top-k relevant chunks from FAISS        │
│  - Generate LLM responses with context              │
│  Output: Natural language answers + code context    │
└─────────────────────────────────────────────────────┘
```

### Technology Stack

- **Code Parsing**: Tree-sitter (Java, Python, TypeScript)
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: GraphCodeBERT, OpenAI text-embedding-3-large
- **LLM Integration**: OpenAI GPT-4 / GPT-3.5
- **Data Processing**: NumPy, Pydantic
- **UI**: Rich (terminal UI), tqdm (progress bars)

---

## 🎥 Demo Video

Watch the complete walkthrough of CodeMind in action:

**[📺 CodeMind Demo - Full Tutorial](https://drive.google.com/file/d/1--L0Ma7qireHlit9fFPDaLoobHcW2-M2/view?usp=sharing)**

> *Click the link above to watch the complete demonstration of the CodeMind platform*

---

## 📁 Project Structure

```
CodeMind/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── Flow diagram.png                       # Architecture diagram
│
├── platform/                              # Core pipeline components
│   ├── 1.parser/                          # Code analysis parser
│   │   ├── ultimate_code_analyzer.py      # Main parser script
│   │   └── output parser/
│   │       └── f.json                     # Analysis output (generated)
│   │
│   ├── 2.chunker/                         # Semantic chunker
│   │   ├── metadata_enriched_chunker.py   # Chunking script
│   │   └── chunk_Data/
│   │       └── chunk                      # Enriched chunks (generated)
│   │
│   ├── 3.embedder/                        # Vector embeddings
│   │   ├── codebert_embedder.py           # Embedding script
│   │   └── vector DB/
│   │       ├── code_index.faiss           # FAISS index (generated)
│   │       └── metadata.npy               # Chunk metadata (generated)
│   │
│   └── 4.retriever/                       # QA & Retrieval
│       └── query_qa_v2.py                 # Interactive QA system
│
└── project/                               # Sample project to analyze
    └── openmrs-client-omod/               # OpenMRS OMOD module
        ├── pom.xml                        # Maven configuration
        └── src/                           # Source code
```

---

## 🚀 Installation

### Prerequisites

- **Python 3.8+** (3.9 or 3.10 recommended)
- **pip** package manager
- **Git** (for cloning repositories to analyze)

### Step 1: Clone the Repository

```bash
git clone https://github.com/saurabh9001/CodeMind-.git
cd CodeMind
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Set Up OpenAI API Key

Create a `.env` file in the project root:

```bash
# .env
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Use OpenRouter instead
# OPENAI_API_KEY=sk-or-v1-your_openrouter_key
```

> **Note**: Get your OpenAI API key from [platform.openai.com](https://platform.openai.com) or use [OpenRouter](https://openrouter.ai) for multi-model access.

---

## 📖 Usage Guide

### Complete Pipeline Workflow

#### Step 1: Parse Your Codebase

Analyze Java source code and extract comprehensive metadata:

```bash
# Navigate to parser directory
cd platform/1.parser

# Run parser on your codebase
python ultimate_code_analyzer.py /path/to/your/java/project

# Or use the sample project included
python ultimate_code_analyzer.py ../../project/openmrs-client-omod
```

**Output**: `output parser/f.json` — Contains complete code analysis including:
- Class and method definitions
- Call graphs (who calls what)
- Spring component annotations
- Database operations
- Scheduled tasks
- Error handling patterns
- Configuration usage

#### Step 2: Create Semantic Chunks

Transform the analysis into enriched, embeddable chunks:

```bash
# Navigate to chunker directory
cd ../2.chunker

# Run chunker on parser output
python metadata_enriched_chunker.py \
  ../1.parser/output\ parser/f.json \
  chunk_Data/enriched_chunks.json
```

**Output**: `chunk_Data/enriched_chunks.json` — Contains:
- Method-level chunks with semantic descriptions
- Complexity and risk metrics
- Caller/callee relationships
- Dependencies and metadata

#### Step 3: Generate Embeddings & Build Index

Create vector embeddings and FAISS index:

```bash
# Navigate to embedder directory
cd ../3.embedder

# Generate embeddings using OpenAI (recommended)
python codebert_embedder.py \
  ../2.chunker/chunk_Data/enriched_chunks.json \
  vector\ DB/ \
  --use-openai

# Or use GraphCodeBERT locally (no API key needed, but slower)
python codebert_embedder.py \
  ../2.chunker/chunk_Data/enriched_chunks.json \
  vector\ DB/
```

**Output**: 
- `vector DB/code_index.faiss` — FAISS similarity search index
- `vector DB/metadata.npy` — Chunk metadata for retrieval

#### Step 4: Query Your Codebase

Start the interactive QA system:

```bash
# Navigate to retriever directory
cd ../4.retriever

# Launch interactive QA
python query_qa_v2.py
```

**Example queries**:
```
> What does the AccessionDiff class do?
> How is patient data synchronized with OpenElis?
> Which methods handle database transactions?
> Show me all scheduled tasks in the codebase
> What error handling patterns are used?
```

---

## 🔧 Pipeline Components

### 1. Parser (`ultimate_code_analyzer.py`)

**Purpose**: Deep static analysis of Java codebases

**Features**:
- Tree-sitter AST parsing
- Method and class extraction
- Complete call graph construction
- Spring Framework component detection
- Database operation tracking
- Scheduled task identification
- Error handling pattern recognition

**Configuration**:
```python
# Edit in ultimate_code_analyzer.py (lines 28-30)
DEFAULT_REPO_PATH = "/path/to/your/java/project"
DEFAULT_OUTPUT_FILE = "/path/to/output/f.json"
```

**Output Format** (`f.json`):
```json
{
  "method_call_graph": {
    "ClassName.methodName": {
      "calls": ["OtherClass.otherMethod"],
      "called_by": ["CallerClass.callerMethod"]
    }
  },
  "class_to_file_mapping": {},
  "spring_components": {},
  "database_operations": [],
  "scheduled_tasks": [],
  "error_handling": []
}
```

### 2. Chunker (`metadata_enriched_chunker.py`)

**Purpose**: Convert analysis into semantic, embeddable chunks

**Features**:
- Method-level chunking
- Rich metadata injection
- Semantic description generation
- Complexity and risk scoring
- Context preservation

**Usage**:
```bash
python metadata_enriched_chunker.py <input_json> <output_chunks>
```

**Output Format** (`enriched_chunks.json`):
```json
[
  {
    "id": "ClassName.methodName",
    "semantic_text": "Spring Service with complexity=15 risk=high...",
    "file_path": "/path/to/File.java",
    "spring_component_type": "Service",
    "complexity": 15,
    "risk_level": "high",
    "calls": ["Method1", "Method2"],
    "called_by": ["CallerMethod"],
    "database_operations": [...],
    "error_handling": [...]
  }
]
```

### 3. Embedder (`codebert_embedder.py`)

**Purpose**: Generate vector embeddings and build FAISS index

**Embedding Options**:

1. **OpenAI (Recommended)** — Fast, high-quality, requires API key
   ```bash
   python codebert_embedder.py <chunks_path> <output_dir> --use-openai
   ```
   - Model: `text-embedding-3-large`
   - Dimensions: 3072
   - Cost: ~$0.13 per 1M tokens

2. **GraphCodeBERT (Local)** — Free, code-aware, slower
   ```bash
   python codebert_embedder.py <chunks_path> <output_dir>
   ```
   - Model: `microsoft/graphcodebert-base`
   - Dimensions: 768
   - Requires: `transformers`, `torch`

**Output**:
- `code_index.faiss` — FAISS L2 index
- `metadata.npy` — NumPy array of chunk metadata

### 4. Retriever (`query_qa_v2.py`)

**Purpose**: Interactive QA system with RAG

**Features**:
- Natural language query embedding
- Top-k similarity retrieval from FAISS
- Context-aware prompt construction
- LLM-powered answer generation
- Rich metadata display

**Configuration**:
```python
# In query_qa_v2.py
k = 5  # Number of chunks to retrieve
model = "gpt-4"  # Or "gpt-3.5-turbo"
```

**Query Flow**:
1. User enters natural language query
2. Query is embedded using same model as chunks
3. FAISS retrieves top-k similar chunks
4. Context is built from retrieved chunks + metadata
5. LLM generates answer based on context
6. Answer displayed with source references

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Required: OpenAI API Key
OPENAI_API_KEY=sk-xxx

# Optional: OpenRouter (for multi-model access)
# OPENAI_API_KEY=sk-or-v1-xxx
```

### Parser Configuration

Edit `platform/1.parser/ultimate_code_analyzer.py`:

```python
# Lines ~28-30
DEFAULT_REPO_PATH = "/path/to/your/project"
DEFAULT_OUTPUT_FILE = "/path/to/output/f.json"
```

### Retriever Configuration

Edit `platform/4.retriever/query_qa_v2.py`:

```python
# Number of chunks to retrieve
k = 5

# LLM model for answers
model = "gpt-4"  # or "gpt-3.5-turbo"
```

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/CodeMind-.git
cd CodeMind

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

thank you---------------------


