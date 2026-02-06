"""
Streamlit App for Code Retriever with Project File Browser
Interactive UI to query the codebase and see retrieved context
"""
import streamlit as st
import sys
from pathlib import Path
import json
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from retriever_raw import (
    load_index_and_metadata,
    retrieve_top_k,
    format_retrieved_context,
    build_llm_prompt
)

# Project path
PROJECT_PATH = "/Users/home/Desktop/new/CodeMind-/project"

# Page config
st.set_page_config(
    page_title="CodeMind Retriever",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with modern design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main Header */
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        animation: fadeInDown 0.8s ease-out;
    }
    
    .sub-header {
        font-size: 1.3rem;
        color: #64748b;
        margin-bottom: 2.5rem;
        animation: fadeInUp 0.8s ease-out;
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Badge */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    .badge-primary {
        background-color: #dbeafe;
        color: #1e40af;
    }
    
    .badge-success {
        background-color: #d1fae5;
        color: #065f46;
    }
    
    .badge-warning {
        background-color: #fef3c7;
        color: #92400e;
    }
    
    .badge-danger {
        background-color: #fee2e2;
        color: #991b1b;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 0.75rem;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'index' not in st.session_state:
    with st.spinner("Loading FAISS index and metadata..."):
        st.session_state.index, st.session_state.metadata = load_index_and_metadata()
        st.session_state.loaded = True

if 'selected_file' not in st.session_state:
    st.session_state.selected_file = None
    
if 'show_file_content' not in st.session_state:
    st.session_state.show_file_content = False

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    # Output format
    output_format = st.selectbox(
        "Output Format",
        ["text", "prompt", "detailed", "json"],
        index=0,
        help="Choose how to display the retrieved context"
    )
    
    # Number of results
    top_k = st.slider(
        "Number of Results",
        min_value=1,
        max_value=20,
        value=5,
        help="How many code chunks to retrieve"
    )
    
    st.divider()
    
    # Index info with styled metrics
    if st.session_state.loaded:
        st.markdown("### 📊 Index Status")
        st.success("✅ Ready")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Vectors", f"{st.session_state.index.ntotal:,}", help="Total indexed vectors")
        with col2:
            st.metric("Dimension", st.session_state.index.d, help="Vector dimension size")
        
        st.metric("Chunks", f"{len(st.session_state.metadata):,}", help="Total code chunks")
    
    st.divider()
    
    # Format descriptions
    with st.expander("📖 Format Guide", expanded=False):
        st.markdown("""
        **🎯 text**: Complete LLM prompt with system message
        
        **📋 prompt**: Full API call structure (most detailed)
        
        **🔍 detailed**: All metadata fields expanded
        
        **📦 json**: Structured JSON output
        """)
    
    # Example queries
    with st.expander("💡 Example Queries", expanded=False):
        examples = [
            "How many APIs are there?",
            "What are scheduled tasks?",
            "Show me database operations",
            "High complexity methods",
            "Patient data migration",
            "Error handling patterns"
        ]
        for example in examples:
            if st.button(example, key=f"ex_{example}", use_container_width=True):
                st.session_state.query_input = example
                st.rerun()
    
    st.divider()
    
    # Footer
    st.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 0.85rem; margin-top: 2rem;">
        <p><strong>CodeMind Platform</strong></p>
        <p>Powered by FAISS + OpenAI</p>
    </div>
    """, unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🔍 CodeMind Retriever</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Query your codebase and browse project files</div>', unsafe_allow_html=True)

# Create main layout with project tree and content
left_col, right_col = st.columns([1, 3])

# Left column - Project File Tree
with left_col:
    st.markdown("### 📂 Project Files")
    st.caption(f"**Path:** `.../{os.path.basename(PROJECT_PATH)}`")
    
    # Search filter for files
    file_search = st.text_input("🔍 Filter files", placeholder="e.g., Mapper, Patient")
    
    st.divider()
    
    # Display file tree in scrollable container
    with st.container():
        if os.path.exists(PROJECT_PATH):
            try:
                # Get root level items
                root_items = sorted(os.listdir(PROJECT_PATH))
                
                for item in root_items:
                    item_path = os.path.join(PROJECT_PATH, item)
                    
                    # Apply search filter
                    if file_search and file_search.lower() not in item.lower():
                        # Check if any file inside matches
                        has_match = False
                        if os.path.isdir(item_path):
                            for root, dirs, files in os.walk(item_path):
                                if any(file_search.lower() in f.lower() for f in files):
                                    has_match = True
                                    break
                        if not has_match:
                            continue
                    
                    if os.path.isdir(item_path) and item not in {'.git', '__pycache__', 'node_modules', '.idea'}:
                        with st.expander(f"📁 {item}", expanded=False):
                            # Show subdirectories and files
                            def show_directory_contents(dir_path, level=0, max_level=2):
                                if level >= max_level:
                                    return
                                try:
                                    items = sorted(os.listdir(dir_path))
                                    dirs = [i for i in items if os.path.isdir(os.path.join(dir_path, i)) 
                                           and i not in {'.git', '__pycache__', 'node_modules', '.idea', 'target', 'build', '.settings', 'bin'}]
                                    files = [i for i in items if os.path.isfile(os.path.join(dir_path, i))]
                                    
                                    # Show directories first
                                    for d in dirs:
                                        if file_search and file_search.lower() not in d.lower():
                                            # Check if has matching files inside
                                            sub_path = os.path.join(dir_path, d)
                                            has_match = False
                                            for root, _, files_in_dir in os.walk(sub_path):
                                                if any(file_search.lower() in f.lower() for f in files_in_dir):
                                                    has_match = True
                                                    break
                                            if not has_match:
                                                continue
                                        
                                        sub_path = os.path.join(dir_path, d)
                                        with st.expander(f"{'  ' * level}📁 {d}", expanded=False):
                                            show_directory_contents(sub_path, level + 1, max_level)
                                    
                                    # Show files
                                    for f in files:
                                        if file_search and file_search.lower() not in f.lower():
                                            continue
                                        
                                        file_path = os.path.join(dir_path, f)
                                        icon = "📄"
                                        if f.endswith('.java'):
                                            icon = "☕"
                                        elif f.endswith(('.xml', '.properties')):
                                            icon = "⚙️"
                                        elif f.endswith(('.md', '.txt')):
                                            icon = "📝"
                                        elif f.endswith('.sql'):
                                            icon = "🗄️"
                                        
                                        if st.button(f"{'  ' * level}{icon} {f}", key=file_path, use_container_width=True):
                                            st.session_state.selected_file = file_path
                                            st.session_state.show_file_content = True
                                            st.rerun()
                                            
                                except PermissionError:
                                    st.caption("⚠️ Permission denied")
                            
                            show_directory_contents(item_path)
                    elif os.path.isfile(item_path):
                        # Root level file
                        icon = "📄"
                        if item.endswith('.java'):
                            icon = "☕"
                        elif item.endswith(('.xml', '.properties')):
                            icon = "⚙️"
                        elif item.endswith(('.md', '.txt')):
                            icon = "📝"
                        
                        if st.button(f"{icon} {item}", key=item_path, use_container_width=True):
                            st.session_state.selected_file = item_path
                            st.session_state.show_file_content = True
                            st.rerun()
                            
            except Exception as e:
                st.error(f"Error reading directory: {e}")
        else:
            st.error("Project path not found")
    
    # Show selected file info
    if st.session_state.selected_file:
        st.divider()
        st.caption("**Selected:**")
        st.code(os.path.basename(st.session_state.selected_file), language="text")
        if st.button("❌ Clear", use_container_width=True):
            st.session_state.selected_file = None
            st.session_state.show_file_content = False
            st.rerun()

# Right column - Main Content
with right_col:
    # Show file content if selected
    if st.session_state.show_file_content and st.session_state.selected_file:
        file_path = st.session_state.selected_file
        
        st.markdown(f"### 📄 {os.path.basename(file_path)}")
        st.caption(f"**Path:** `{file_path}`")
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Show file stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Lines", len(content.splitlines()))
            with col2:
                st.metric("Characters", len(content))
            with col3:
                st.metric("Size", f"{len(content.encode('utf-8')) / 1024:.1f} KB")
            
            st.divider()
            
            # Display content with syntax highlighting
            language = "text"
            if file_path.endswith('.java'):
                language = "java"
            elif file_path.endswith('.xml'):
                language = "xml"
            elif file_path.endswith('.properties'):
                language = "properties"
            elif file_path.endswith('.sql'):
                language = "sql"
            elif file_path.endswith('.md'):
                language = "markdown"
            
            st.code(content, language=language, line_numbers=True)
            
            # Action buttons
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📥 Download File",
                    data=content,
                    file_name=os.path.basename(file_path),
                    mime="text/plain",
                    use_container_width=True
                )
            with col2:
                if st.button("🔍 Search Related Code", use_container_width=True):
                    st.session_state.query_input = f"Show me code related to {os.path.basename(file_path).replace('.java', '')}"
                    st.session_state.show_file_content = False
                    st.rerun()
            
        except Exception as e:
            st.error(f"Error reading file: {e}")
    
    else:
        # Original search interface
        col1, col2 = st.columns([2, 1])

        with col1:
            # Query input
            query = st.text_input(
                "Enter your question:",
                placeholder="e.g., How many APIs are there?",
                key="query_input"
            )

        with col2:
            # Search button
            st.write("")  # Spacing
            search_button = st.button("🔍 Search", type="primary", use_container_width=True)

        # Process search
        if search_button and query:
            with st.spinner("🔍 Searching through 7,942 code chunks..."):
                # Retrieve chunks
                results = retrieve_top_k(
                    st.session_state.index,
                    st.session_state.metadata,
                    query,
                    k=top_k,
                    use_openai_embed=True
                )
                
                # Display metrics
                st.divider()
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{len(results)}</div>
                        <div class="metric-label">Chunks Retrieved</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    avg_distance = sum(r['distance'] for r in results) / len(results)
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{avg_distance:.3f}</div>
                        <div class="metric-label">Avg Distance</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    best_match = results[0]['distance']
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{best_match:.3f}</div>
                        <div class="metric-label">Best Match</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    total_chars = sum(len(r['chunk'].get('semantic_text', '')) for r in results)
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{total_chars:,}</div>
                        <div class="metric-label">Context Chars</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.divider()
                
                # Tabs
                tab1, tab2, tab3 = st.tabs(["📝 Formatted Output", "🎯 Quick View", "📊 Chunk Details"])
                
                with tab1:
                    st.markdown("### 📄 Retrieved Context")
                    
                    output = format_retrieved_context(query, results, output_format=output_format)
                    
                    # Show the user's question above the output
                    st.markdown(f"**User Question:**\n\n`{query}`\n\n---")
                    
                    st.info("💡 **Copy this context and paste it into ChatGPT along with your question**")
                    
                    if output_format == 'json':
                        st.json(json.loads(output))
                    else:
                        st.code(output, language="text", line_numbers=True)
                    
                    # Action buttons
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.download_button(
                            label="📥 Download",
                            data=output,
                            file_name=f"context_{output_format}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    with col2:
                        # Create a copyable text area
                        if st.button("📋 Show Copyable", use_container_width=True, help="Show text for easy copy-paste"):
                            st.session_state.show_copyable = True
                    with col3:
                        # Open ChatGPT button
                        st.link_button(
                            label="🤖 Open ChatGPT",
                            url="https://chatgpt.com",
                            use_container_width=True
                        )
                    with col4:
                        if st.button("🔄 New Search", use_container_width=True):
                            st.session_state.show_copyable = False
                            st.rerun()
                    
                    # Show copyable text area if button clicked
                    if st.session_state.get('show_copyable', False):
                        st.markdown("---")
                        st.markdown("**📋 Select all text below (Cmd+A / Ctrl+A) and copy (Cmd+C / Ctrl+C):**")
                        st.text_area(
                            "Copyable Text",
                            f"User Question:\n{query}\n\n---\n{output}",
                            height=300,
                            label_visibility="collapsed",
                            key="copyable_output"
                        )
                        st.markdown("""
                        **Next Steps:**
                        1. ✅ Select all text above
                        2. 📋 Copy it (Cmd+C or Ctrl+C)
                        3. 🤖 Click "Open ChatGPT" button above
                        4. 📝 Paste the context and ask your question
                        """)
                
                with tab2:
                    st.markdown("### 🎯 Quick View")
                    
                    for r in results:
                        chunk = r['chunk']
                        risk_level = chunk.get('risk_level', 'unknown').lower()
                        risk_badge = {
                            'high': 'badge-danger',
                            'medium': 'badge-warning',
                            'low': 'badge-success',
                            'unknown': 'badge-primary'
                        }.get(risk_level, 'badge-primary')
                        
                        with st.expander(f"**#{r['rank']} {chunk['id']}** · Distance: {r['distance']:.4f}", expanded=(r['rank'] <= 2)):
                            st.markdown(f"""
                            <div>
                                <span class="badge {risk_badge}">{chunk.get('risk_level', 'unknown').upper()} RISK</span>
                                <span class="badge badge-primary">{chunk.get('spring_component_type', 'unknown')}</span>
                                <span class="badge badge-primary">Complexity: {chunk.get('complexity', 0)}</span>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.write("")
                            
                            if chunk.get('file_path'):
                                st.code(chunk.get('file_path'), language="text")
                            
                            st.markdown("**📝 Description:**")
                            st.info(chunk.get('semantic_text', 'No description'))
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("📞 Calls", len(chunk.get('calls', [])))
                            with col2:
                                st.metric("📲 Called By", len(chunk.get('called_by', [])))
                            with col3:
                                st.metric("🗄️ DB Ops", len(chunk.get('database_operations', [])))
                            with col4:
                                st.metric("🔗 Dependencies", len(chunk.get('class_dependencies', [])))
                
                with tab3:
                    st.markdown("### 📊 Detailed Analysis")
                    
                    for r in results:
                        chunk = r['chunk']
                        st.markdown(f"#### 🔹 Chunk {r['rank']}: `{chunk['id']}`")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**📞 Method Calls:**")
                            calls = chunk.get('calls', [])
                            if calls:
                                for i, call in enumerate(calls[:10], 1):
                                    st.markdown(f"`{i}.` {call}")
                                if len(calls) > 10:
                                    st.caption(f"... and {len(calls) - 10} more")
                            else:
                                st.caption("No method calls")
                            
                            st.write("")
                            st.markdown("**🔗 Class Dependencies:**")
                            deps = chunk.get('class_dependencies', [])
                            if deps:
                                for i, dep in enumerate(deps[:10], 1):
                                    st.markdown(f"`{i}.` {dep}")
                                if len(deps) > 10:
                                    st.caption(f"... and {len(deps) - 10} more")
                            else:
                                st.caption("No dependencies")
                        
                        with col2:
                            st.markdown("**📲 Called By:**")
                            called_by = chunk.get('called_by', [])
                            if called_by:
                                for i, caller in enumerate(called_by[:10], 1):
                                    st.markdown(f"`{i}.` {caller}")
                                if len(called_by) > 10:
                                    st.caption(f"... and {len(called_by) - 10} more")
                            else:
                                st.caption("Not called by any method")
                            
                            st.write("")
                            if chunk.get('database_operations'):
                                st.markdown("**🗄️ Database Operations:**")
                                for i, op in enumerate(chunk.get('database_operations', []), 1):
                                    st.markdown(f"`{i}.` {op}")
                            
                            if chunk.get('is_scheduled'):
                                st.markdown("**⏰ Scheduled Task Info:**")
                                st.json(chunk.get('scheduled_info', {}))
                        
                        st.divider()
        
        elif query and not search_button:
            st.info("👆 Click the Search button to query the codebase")
        
        else:
            # Compact welcome message
            st.info("💡 **Quick Start:** Browse files on the left or enter a question above to search through 7,942 code chunks")
            
            st.markdown("### 📊 Codebase Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            stats = [
                ("7,942", "Total Methods"),
                ("448", "Java Classes"),
                ("15", "External APIs"),
                ("104", "DB Operations")
            ]
            
            for (value, label), col in zip(stats, [col1, col2, col3, col4]):
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{value}</div>
                        <div class="metric-label">{label}</div>
                    </div>
                    """, unsafe_allow_html=True)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    Built with ❤️ using Streamlit | CodeMind Platform
</div>
""", unsafe_allow_html=True)
