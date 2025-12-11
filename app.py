"""Main Streamlit application for OpenAI chat with vector store RAG."""
import json
from datetime import datetime
from typing import List, Dict

import streamlit as st

from openai_client import OpenAIClient
from config import Config
from logger import api_error_logger

# Page configuration
st.set_page_config(
    page_title="OpenAI Chat with RAG",
    page_icon="💬",
    layout="wide"
)

# Initialize session state
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "messages": [],
        "message_sources": {},  # Map message index to sources
        "debug_mode": False,
        "current_page": "chat",
        "research_results": [],
        "relevance_threshold": 0.55,
        "selected_snippets": [],
        "filters": {"client": None, "year": None, "topic": None},
        "current_sources": [],
    }
    
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default
    
    # Initialize OpenAI client
    if "openai_client" not in st.session_state:
        try:
            st.session_state.openai_client = OpenAIClient()
        except ValueError as e:
            st.error(f"Configuration Error: {str(e)}")
            st.stop()

init_session_state()


def display_source_expander(sources: List[Dict], title_prefix: str = ""):
    """Display sources in an expander with consistent formatting."""
    if not sources:
        return
    
    with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
        for num, source in enumerate(sources, 1):
            filename = source.get('filename', f'Document {num}')
            display_name = filename.split('/')[-1] if '/' in filename else filename
            
            st.markdown(f"**{num}. {display_name}**")
            
            # Show snippet/quote if available
            snippet = source.get('snippet') or source.get('content', '')
            if snippet:
                max_len = 300
                display_text = snippet[:max_len] + "..." if len(snippet) > max_len else snippet
                st.markdown(display_text)
            
            # Show relevance score if available
            score = source.get('score')
            if score is not None:
                st.caption(f"Relevance: {score:.3f}")
            
            if num < len(sources):
                st.divider()


def log_query(question: str, answer: str, sources: List[Dict], filters: Dict):
    """Log query to JSONL file for debugging."""
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "top_sources": [
                {"filename": src.get('filename', 'Unknown'), "score": src.get('score')}
                for src in sources[:5]
            ],
            "filters": filters
        }
        
        with open("query_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception:
        pass  # Don't fail if logging fails


# Sidebar
with st.sidebar:
    st.header("Navigation")
    
    page_options = {
        "about": "💡 About",
        "chat": "💬 Chat with BD Knowledge Base",
        "vector_search": "🔍 Vector Search"
    }
    
    selected_page = st.radio(
        "Select Page",
        options=list(page_options.keys()),
        format_func=lambda x: page_options[x],
        index=list(page_options.keys()).index(st.session_state.current_page)
    )
    
    if selected_page != st.session_state.current_page:
        st.session_state.current_page = selected_page
        st.rerun()
    
    st.divider()
    
    # Settings
    with st.expander("⚙️ Settings", expanded=False):
        st.info(f"**Model:** {Config.OPENAI_MODEL}")
        
        if st.session_state.debug_mode:
            vs_id = Config.OPENAI_VECTOR_STORE_ID or "Not configured"
            st.info(f"**Vector Store ID:** `{vs_id}`")
            
            client = st.session_state.openai_client
            api_status = "✅ Responses API" if client.responses_api_available else "❌ Direct Search Only"
            st.info(f"**API Mode:** {api_status}")
            
            if client.last_api_used:
                st.caption(f"**Last Used:** `{client.last_api_used}`")
        else:
            st.info("Vector store configured ✓")
        
        st.divider()
        
        st.session_state.debug_mode = st.checkbox(
            "🐛 Debug Mode",
            value=st.session_state.debug_mode,
            help="Show technical details"
        )
        
        st.divider()
        
        st.session_state.relevance_threshold = st.slider(
            "📊 Min Relevance Score",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.relevance_threshold,
            step=0.05,
            help="Filter sources below this threshold"
        )
        
        st.divider()
        
        if st.button("🗑️ Clear Chat History", type="secondary", use_container_width=True):
            st.session_state.messages = []
            st.session_state.message_sources = {}
            st.session_state.research_results = []
            st.session_state.current_sources = []
            st.session_state.selected_snippets = []
            st.rerun()


# Page routing
if st.session_state.current_page == "about":
    st.title("About")
    st.markdown("""
    ## Business Development Resource Library
    
    This application provides intelligent access to your business development knowledge base 
    using advanced AI and vector search technology.
    
    ### Features
    
    #### Chat with BD Knowledge Base
    Ask questions in natural language and get comprehensive answers backed by your documents:
    1. Your question is processed and searched against the vector store
    2. Relevant document chunks are retrieved using semantic search
    3. The AI generates a response using the retrieved context
    4. Sources are displayed so you can verify the information
    
    #### Vector Search
    Search the database directly for documents without AI generation.
    Returns the most relevant document chunks for your query.
    
    ### How to Use
    - **Chat Page**: Type questions and receive AI-generated answers with sources
    - **Vector Search Page**: Enter queries to explore documents directly
    - **Settings**: Configure debug mode and relevance thresholds
    """)

elif st.session_state.current_page == "chat":
    st.title("Chat with BD Knowledge Base")
    st.markdown("Ask questions and get answers enhanced with context from your vector store.")
    
    # Display chat history
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display sources for assistant messages
            if message["role"] == "assistant" and idx in st.session_state.message_sources:
                sources = st.session_state.message_sources[idx]
                display_source_expander(sources)
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            sources_placeholder = st.empty()
            
            with st.spinner("Querying vector store and generating response..."):
                try:
                    conversation_history = st.session_state.messages[:-1]
                    
                    response_text, sources = st.session_state.openai_client.get_rag_response(
                        user_query=prompt,
                        conversation_history=conversation_history,
                        min_relevance_score=st.session_state.relevance_threshold
                    )
                    
                    # Debug info
                    client = st.session_state.openai_client
                    if st.session_state.debug_mode and client.last_filtered_count > 0:
                        st.info(f"🔍 Filtered {client.last_filtered_count} source(s) below threshold")
                    
                    # Display response
                    message_placeholder.markdown(response_text)
                    
                    # Display sources
                    with sources_placeholder.container():
                        display_source_expander(sources)
                    
                    # Log query
                    log_query(prompt, response_text, sources, st.session_state.filters)
                    
                    # Store message and sources
                    message_idx = len(st.session_state.messages)
                    st.session_state.message_sources[message_idx] = sources
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    
                except ValueError as e:
                    error_msg = f"⚠️ Configuration Error: {str(e)}"
                    api_error_logger.error(f"Config error: {str(e)}")
                    message_placeholder.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    api_error_logger.error(f"API error: {str(e)}")
                    message_placeholder.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

elif st.session_state.current_page == "vector_search":
    st.title("Vector Search")
    st.markdown("Search the database directly for documents.")
    st.divider()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🔍 Search & Filters")
        
        # Filters
        st.markdown("**Filters:**")
        filters = {
            "client": st.text_input("Client", value=st.session_state.filters.get("client") or ""),
            "year": st.text_input("Year", value=st.session_state.filters.get("year") or ""),
            "topic": st.text_input("Topic", value=st.session_state.filters.get("topic") or ""),
        }
        st.session_state.filters = filters
        
        # Search
        query = st.text_input("Search query:", placeholder="e.g., fiscal management")
        
        if st.button("🔍 Search", type="primary", use_container_width=True):
            if query:
                with st.spinner("Searching..."):
                    try:
                        active_filters = {k: v for k, v in filters.items() if v}
                        st.session_state.research_results = st.session_state.openai_client.search_vectors(
                            query=query,
                            top_k=50,
                            filters=active_filters or None,
                            min_relevance_score=st.session_state.relevance_threshold
                        )
                        st.session_state.selected_snippets = []
                    except Exception as e:
                        api_error_logger.error(f"Search failed: {str(e)}")
                        st.error(f"Search failed: {str(e)}")
                        st.session_state.research_results = []
            else:
                st.warning("Please enter a search query")
        
        # Summarize selected
        if st.session_state.selected_snippets and st.session_state.research_results:
            st.divider()
            st.subheader("📝 Summary")
            
            if st.button("Summarize Selected", type="secondary", use_container_width=True):
                selected = [
                    st.session_state.research_results[i]
                    for i in st.session_state.selected_snippets
                    if i < len(st.session_state.research_results)
                ]
                
                if selected:
                    context = "\n\n---\n\n".join([
                        f"From {s.get('filename', 'Document')}:\n{s.get('snippet', '')}"
                        for s in selected
                    ])
                    
                    messages = [
                        {"role": "system", "content": "Summarize the provided document snippets."},
                        {"role": "user", "content": f"Please summarize:\n\n{context}"}
                    ]
                    
                    with st.spinner("Generating summary..."):
                        try:
                            response = st.session_state.openai_client.get_chat_completion(
                                messages=messages, context=None, stream=False
                            )
                            if response.choices:
                                st.markdown(response.choices[0].message.content)
                        except Exception as e:
                            api_error_logger.error(f"Summary failed: {str(e)}")
                            st.error(f"Failed to summarize: {str(e)}")
    
    with col2:
        st.subheader("📚 Search Results")
        
        results = st.session_state.research_results
        
        if results:
            st.success(f"✅ Found {len(results)} result(s)")
            st.divider()
            
            # Group by filename
            by_file = {}
            for idx, result in enumerate(results):
                filename = result.get('filename', f'Document {idx + 1}')
                if filename not in by_file:
                    by_file[filename] = []
                by_file[filename].append((idx, result))
            
            # Display grouped results
            for filename, items in by_file.items():
                display_name = filename.split('/')[-1] if '/' in filename else filename
                
                with st.expander(f"📄 {display_name} ({len(items)} snippet(s))", expanded=False):
                    for idx, result in items:
                        # Selection checkbox
                        selected = st.checkbox(
                            f"Select snippet {idx + 1}",
                            value=idx in st.session_state.selected_snippets,
                            key=f"snippet_{idx}"
                        )
                        
                        if selected and idx not in st.session_state.selected_snippets:
                            st.session_state.selected_snippets.append(idx)
                        elif not selected and idx in st.session_state.selected_snippets:
                            st.session_state.selected_snippets.remove(idx)
                        
                        # Display snippet
                        snippet = result.get('snippet') or result.get('content', '')
                        if snippet:
                            st.markdown(snippet)
                        
                        # Show score
                        score = result.get('score')
                        if score is not None:
                            st.caption(f"Relevance: {score:.3f}")
                        
                        if idx < items[-1][0]:
                            st.divider()
        
        elif query:
            st.info("No documents found. Try a different search query.")
        else:
            st.info("Enter a search query and click Search to explore documents.")
    
    # Debug panel
    if st.session_state.debug_mode:
        with st.expander("🐛 Debug Information", expanded=False):
            st.subheader("Session State")
            st.json({
                "total_messages": len(st.session_state.messages),
                "messages_with_sources": len(st.session_state.message_sources),
                "research_results_count": len(st.session_state.research_results),
                "vector_store_id": Config.OPENAI_VECTOR_STORE_ID,
                "model": Config.OPENAI_MODEL
            })
            
            st.subheader("API Usage")
            client = st.session_state.openai_client
            st.json({
                "Responses API Available": client.responses_api_available,
                "Last API Used": client.last_api_used or "None",
                "Last Error": client.last_error or "None"
            })
