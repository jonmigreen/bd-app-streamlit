"""Main Streamlit application for OpenAI chat with vector store RAG."""
import os
from typing import List, Dict

import streamlit as st
from openai_client import OpenAIClient
from config import Config
from logger import api_error_logger

# Verify we're using the correct OpenAI package (not system Python)
try:
    import openai
    openai_path = os.path.dirname(openai.__file__)
    openai_version = openai.__version__
    
    # Check if we're using system Python's packages
    if '/Library/Frameworks/Python.framework' in openai_path:
        st.error(f"⚠️ **Wrong Python Environment Detected!**\n\n"
                 f"Streamlit is using the system Python (OpenAI {openai_version}) instead of your virtual environment.\n\n"
                 f"**Solution:**\n"
                 f"1. Stop this app (Ctrl+C)\n"
                 f"2. Activate your virtual environment: `source venv/bin/activate`\n"
                 f"3. Verify: `which python` should show `.../venv/bin/python`\n"
                 f"4. Run Streamlit again: `streamlit run app.py`\n\n"
                 f"Current OpenAI location: `{openai_path}`")
        st.stop()
    
    # Verify we have the correct OpenAI version (1.x, not 2.x)
    if openai_version.startswith('2.'):
        st.error(f"⚠️ **Incompatible OpenAI Version!**\n\n"
                 f"OpenAI SDK version {openai_version} is installed, but this app requires version 1.x.\n\n"
                 f"**Solution:**\n"
                 f"1. Activate your virtual environment: `source venv/bin/activate`\n"
                 f"2. Reinstall: `pip install 'openai>=1.12.0,<2.0.0'`\n"
                 f"3. Restart Streamlit")
        st.stop()
except Exception as e:
    st.error(f"Error checking OpenAI installation: {str(e)}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="OpenAI Chat with RAG",
    page_icon="💬",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "message_sources" not in st.session_state:
    st.session_state.message_sources = {}  # Map message index to sources

if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False

if "current_page" not in st.session_state:
    st.session_state.current_page = "chat"  # Options: "about", "chat", "vector_search"

if "research_mode" not in st.session_state:
    st.session_state.research_mode = False

if "research_results" not in st.session_state:
    st.session_state.research_results = []

if "relevance_threshold" not in st.session_state:
    st.session_state.relevance_threshold = 0.55  # Default: middle of 0.5-0.6 range

if "selected_snippets" not in st.session_state:
    st.session_state.selected_snippets = []  # List of selected snippet indices

if "filters" not in st.session_state:
    st.session_state.filters = {"client": None, "year": None, "topic": None}

if "current_sources" not in st.session_state:
    st.session_state.current_sources = []  # Sources for current question

if "openai_client" not in st.session_state:
    try:
        st.session_state.openai_client = OpenAIClient()
    except ValueError as e:
        st.error(f"Configuration Error: {str(e)}")
        st.stop()

# Helper function to format response with in-text citations
def format_response_with_citations(response_text: str, sources: List[Dict]) -> tuple[str, Dict[int, Dict]]:
    """
    Insert citation markers into response text based on citation indices.
    Returns formatted text and citation mapping (citation_number -> source info).
    """
    if not sources:
        return response_text, {}
    
    # Extract citation quotes from sources
    citation_quotes = []
    for source in sources:
        citation_quotes_list = source.get('metadata', {}).get('citation_quotes', [])
        if citation_quotes_list:
            for quote in citation_quotes_list:
                if quote.get('start_index') is not None and quote.get('end_index') is not None:
                    citation_quotes.append({
                        'start_index': quote['start_index'],
                        'end_index': quote['end_index'],
                        'filename': quote.get('filename', source.get('filename', 'Unknown')),
                        'quote': quote.get('quote', ''),
                        'file_id': quote.get('file_id', source.get('file_id'))
                    })
    
    if not citation_quotes:
        return response_text, {}
    
    # Sort citations by start_index in reverse order (to insert from end)
    citation_quotes.sort(key=lambda x: x['start_index'], reverse=True)
    
    # Build citation mapping and insert markers
    formatted_text = response_text
    citation_map = {}
    citation_number = 1
    
    for citation in citation_quotes:
        start_idx = citation['start_index']
        end_idx = citation['end_index']
        
        # Insert citation marker [1], [2], etc. after the cited text
        # Validate indices are within bounds (start_idx used for validation)
        if start_idx is not None and end_idx is not None and 0 <= start_idx <= end_idx <= len(formatted_text):
            marker = f"[{citation_number}]"
            formatted_text = formatted_text[:end_idx] + marker + formatted_text[end_idx:]
            
            # Map citation number to source info
            citation_map[citation_number] = {
                'filename': citation['filename'],
                'quote': citation['quote'],
                'file_id': citation.get('file_id')
            }
            citation_number += 1
    
    return formatted_text, citation_map

# Helper function to display sources
def display_sources(sources: List[Dict], message_idx: int, debug_mode: bool = False):
    """Display sources in a collapsible panel."""
    if not sources:
        return
    
    with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
        for idx, source in enumerate(sources):
            with st.container():
                # Display filename or document title (clean, no technical IDs)
                # Try multiple locations for filename
                filename = (
                    source.get('filename') or 
                    source.get('metadata', {}).get('filename') or
                    f'Document {idx + 1}'
                )
                
                # Clean up filename display
                if filename:
                    # Remove path separators and clean up
                    display_name = filename.split('/')[-1].split('\\')[-1]
                    # Remove file extensions for cleaner display (optional)
                    # display_name = display_name.rsplit('.', 1)[0] if '.' in display_name else display_name
                else:
                    display_name = f'Document {idx + 1}'
                
                # Show a more descriptive title
                st.markdown(f"**📄 {display_name}**")
                
                # Display content snippet
                content = source.get('content', '')
                if content:
                    # Truncate long content for display
                    max_length = 500
                    if len(content) > max_length:
                        content_preview = content[:max_length] + "..."
                        st.markdown(content_preview)
                        with st.expander("View full content"):
                            st.markdown(content)
                    else:
                        st.markdown(content)
                
                # Show debug info if enabled
                if debug_mode and '_debug' in source:
                    debug_info = source['_debug']
                    with st.expander("🔧 Debug Info", expanded=False):
                        st.code(f"File ID: {debug_info.get('file_id', 'N/A')}")
                        if debug_info.get('score') is not None:
                            st.code(f"Relevance Score: {debug_info.get('score'):.4f}")
                
                if idx < len(sources) - 1:
                    st.divider()

# Sidebar for navigation and settings (appears on all pages)
with st.sidebar:
    st.header("Navigation")
    
    # Page navigation
    page_options = {
        "about": "💡 About",
        "chat": "💬 Chat with BD Knowledge Base",
        "vector_search": "🔍 Vector Search"
    }
    
    selected_page = st.radio(
        "Select Page",
        options=list(page_options.keys()),
        format_func=lambda x: page_options[x],
        index=list(page_options.keys()).index(st.session_state.current_page) if st.session_state.current_page in page_options else 1
    )
    
    if selected_page != st.session_state.current_page:
        st.session_state.current_page = selected_page
        st.rerun()
    
    st.divider()
    
    # Settings expander
    with st.expander("⚙️ Settings", expanded=False):
        st.info(f"**Model:** {Config.OPENAI_MODEL}")
        
        # Only show vector store ID in debug mode
        if st.session_state.debug_mode:
            vs_id_display = Config.OPENAI_VECTOR_STORE_ID if Config.OPENAI_VECTOR_STORE_ID else "Not configured"
            st.info(f"**Vector Store ID:** `{vs_id_display}`")
            
            # Show API status indicator
            api_status = "✅ Responses API" if st.session_state.openai_client.responses_api_available else "❌ Direct Search Only"
            st.info(f"**API Mode:** {api_status}")
            if st.session_state.openai_client.last_api_used:
                st.caption(f"**Last Used:** `{st.session_state.openai_client.last_api_used}`")
                # Show warning if Responses API is available but fallback was used
                if (st.session_state.openai_client.responses_api_available and 
                    st.session_state.openai_client.last_api_used == "fallback_rag"):
                    error_msg = getattr(st.session_state.openai_client, 'last_error', 'Unknown error')
                    st.warning(f"⚠️ Responses API failed, using fallback. Error: {error_msg[:100]}...")
        else:
            st.info("Vector store configured ✓")
        
        st.divider()
        
        # Debug Mode toggle
        st.session_state.debug_mode = st.checkbox(
            "🐛 Debug Mode", 
            value=st.session_state.debug_mode, 
            help="Show technical details (file IDs, scores, etc.)"
        )
        
        st.divider()
        
        # Relevance threshold control
        st.session_state.relevance_threshold = st.slider(
            "📊 Min Relevance Score",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.relevance_threshold,
            step=0.05,
            help="Filter out sources with relevance scores below this threshold. Sources without scores are always kept."
        )
        
        st.divider()
        
        if st.button("🗑️ Clear Chat History", type="secondary", use_container_width=True):
            st.session_state.messages = []
            st.session_state.message_sources = {}
            st.session_state.research_results = []
            st.session_state.current_sources = []
            st.session_state.selected_snippets = []
            st.rerun()

# Page routing - render based on current_page

# About Page
if st.session_state.current_page == "about":
    st.title("About")
    
    st.markdown("""
    ## Business Development Resource Library
    
    This application provides intelligent access to your business development knowledge base using advanced AI and vector search technology.
    
    ### Features
    
    #### Chat with BD Knowledge Base using Responses API + File Search
    
    Ask questions in natural language and get comprehensive answers backed by your documents. Here's how it works:
    
    1. **Query Processing**: Your question is converted into an embedding (a numerical representation of meaning). The system also generates alternative query variations to improve search coverage.
    
    2. **Vector Search**: The "file_search" tool searches the Vector Store, finding document chunks whose embeddings are closest to your question's embedding. This semantic search goes beyond keyword matching to understand meaning and context.
    
    3. **Context Retrieval**: The most relevant document chunks are extracted and provided to the Large Language Model (LLM) along with your original question.
    
    4. **Answer Generation**: The LLM uses this carefully selected context to generate a precise, well-informed answer with in-text citations pointing to the source documents.
    
    #### Vector Search (Search Database Directly for Documents)
    
    Perform high-precision similarity searches against the knowledge base without invoking an LLM. This mode returns the most relevant document chunks or documents directly, allowing you to explore the source material yourself.
    
    ### How to Use
    
    - **Chat Page**: Type your question in the chat input and receive AI-generated answers with citations
    - **Vector Search Page**: Enter a search query to explore documents directly
    - **Settings**: Access configuration options including debug mode and relevance threshold controls
    
    ### Note on Filters
    
    Filter functionality (Client, Year, Topic) is currently not fully functional and may not work as expected in all scenarios.
    """)

# Chat Page
elif st.session_state.current_page == "chat":
    st.title("Chat with BD Knowledge Base")
    st.markdown("Ask questions and get answers enhanced with context from your vector store.")
    
    # Display chat history
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            # Format message with citations if it's an assistant message with sources
            # Sources are stored at the message's index (set before appending assistant message)
            if message["role"] == "assistant" and idx in st.session_state.message_sources:
                sources = st.session_state.message_sources[idx]
                formatted_content, _ = format_response_with_citations(message["content"], sources)
                st.markdown(formatted_content)
                
                # Display sources in expander
                if sources:
                    with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
                        for num, source in enumerate(sources, 1):
                            filename = source.get('filename', f'Document {num}')
                            display_name = filename.split('/')[-1] if '/' in filename else filename
                            
                            st.markdown(f"**{num}. {display_name}**")
                            
                            # Show snippet/quote if available
                            snippet = source.get('snippet') or source.get('content', '')
                            if snippet:
                                st.markdown(snippet[:300] + "..." if len(snippet) > 300 else snippet)
                            
                            # Show relevance score if available
                            score = source.get('score')
                            if score is not None:
                                st.caption(f"Relevance: {score:.3f}")
                            
                            if num < len(sources):
                                st.divider()
            else:
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            sources_placeholder = st.empty()
            
            with st.spinner("Querying vector store and generating response..."):
                try:
                    # Get conversation history (excluding the current user message for the API call)
                    conversation_history = st.session_state.messages[:-1]
                    
                    # Get RAG-enhanced response with sources
                    response_text, sources = st.session_state.openai_client.get_rag_response(
                        user_query=prompt,
                        conversation_history=conversation_history,
                        min_relevance_score=st.session_state.relevance_threshold
                    )
                    
                    # Debug logging for filtered sources in RAG response
                    if st.session_state.debug_mode and st.session_state.openai_client.last_filtered_count > 0:
                        st.info(f"🔍 Filtered out {st.session_state.openai_client.last_filtered_count} source(s) below relevance threshold ({st.session_state.relevance_threshold:.2f}) in RAG response")
                    
                    # Format response with citations
                    formatted_response, citation_map = format_response_with_citations(response_text, sources)
                    
                    # Logging for debugging: question, answer, top 5 search results
                    try:
                        import json
                        from datetime import datetime
                        
                        log_entry = {
                            "timestamp": datetime.now().isoformat(),
                            "question": prompt,
                            "answer": response_text,
                            "top_sources": [
                                {
                                    "filename": src.get('filename', 'Unknown'),
                                    "score": src.get('score')
                                }
                                for src in sources[:5]
                            ],
                            "filters": st.session_state.filters
                        }
                        
                        # Simple file logging (append mode)
                        try:
                            with open("query_log.jsonl", "a", encoding="utf-8") as f:
                                f.write(json.dumps(log_entry) + "\n")
                        except Exception:
                            # If file write fails, just print (for debugging)
                            if st.session_state.debug_mode:
                                print(f"LOG: {json.dumps(log_entry, indent=2)}")
                    except Exception:
                        pass  # Don't fail if logging fails
                    
                    # Display formatted response with citations
                    message_placeholder.markdown(formatted_response)
                    
                    # Display sources in expander below response
                    if sources:
                        with sources_placeholder.expander(f"📚 Sources ({len(sources)})", expanded=False):
                            # Build numbered list of sources
                            for num, source in enumerate(sources, 1):
                                filename = source.get('filename', f'Document {num}')
                                display_name = filename.split('/')[-1] if '/' in filename else filename
                                
                                st.markdown(f"**{num}. {display_name}**")
                                
                                # Show snippet/quote if available
                                snippet = source.get('snippet') or source.get('content', '')
                                if snippet:
                                    st.markdown(snippet[:300] + "..." if len(snippet) > 300 else snippet)
                                
                                # Show relevance score if available
                                score = source.get('score')
                                if score is not None:
                                    st.caption(f"Relevance: {score:.3f}")
                                
                                if num < len(sources):
                                    st.divider()
                    
                    # Store sources with message index
                    message_idx = len(st.session_state.messages)
                    st.session_state.message_sources[message_idx] = sources
                    
                    # Add assistant response to chat history (store original, not formatted)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    
                except ValueError as e:
                    # Configuration errors
                    error_msg = str(e)
                    api_error_logger.error(f"Configuration error in chat: {error_msg}")
                    error_message = f"⚠️ Configuration Error: {error_msg}\n\nPlease check your .env file and ensure all required variables are set."
                    message_placeholder.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
                    
                except Exception as e:
                    # API or other errors
                    error_msg = str(e)
                    api_error_logger.error(f"API error in chat mode: {error_msg} | Query: {prompt[:100]}...")
                    error_message = f"❌ Error: {error_msg}\n\nPlease try again or check your API configuration."
                    message_placeholder.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})

# Vector Search Page
elif st.session_state.current_page == "vector_search":
    st.title("Vector Search")
    st.markdown("Search the database directly for documents - high-precision similarity searches without LLM.")
    
    st.divider()
    
    # Two-column layout for Research Mode
    research_col1, research_col2 = st.columns([1, 2])
    
    with research_col1:
        st.subheader("🔍 Search & Filters")
        
        # Filter controls
        st.markdown("**Filters:**")
        research_filters = {}
        research_filters["client"] = st.text_input("Client", key="research_client", value=st.session_state.filters.get("client") or "")
        research_filters["year"] = st.text_input("Year", key="research_year", value=st.session_state.filters.get("year") or "")
        research_filters["topic"] = st.text_input("Topic", key="research_topic", value=st.session_state.filters.get("topic") or "")
        
        # Update session state filters
        st.session_state.filters = research_filters
        
        # Research search input
        research_query = st.text_input("Search query:", placeholder="e.g., fiscal management", key="research_query")
        
        if st.button("🔍 Search", type="primary", use_container_width=True):
            if research_query:
                with st.spinner("Searching documents..."):
                    try:
                        st.session_state.research_results = st.session_state.openai_client.search_vectors(
                            query=research_query,
                            top_k=50,
                            filters=research_filters if any(research_filters.values()) else None,
                            min_relevance_score=st.session_state.relevance_threshold
                        )
                    except Exception as e:
                        error_msg = str(e)
                        api_error_logger.error(f"Search failed in research mode: {error_msg} | Query: {research_query[:100]}...")
                        st.error(f"Search failed: {error_msg}")
                        st.session_state.research_results = []
                    
                    # Debug logging for filtered sources
                    if st.session_state.research_results and st.session_state.debug_mode:
                        if st.session_state.research_results and '_debug' in st.session_state.research_results[0]:
                            filtered_count = st.session_state.research_results[0].get('_debug', {}).get('filtered_count', 0)
                            if filtered_count > 0:
                                st.info(f"🔍 Filtered out {filtered_count} source(s) below relevance threshold ({st.session_state.relevance_threshold:.2f})")
            else:
                st.warning("Please enter a search query")
        
        # Summary area (left pane)
        if st.session_state.selected_snippets and st.session_state.research_results:
            st.divider()
            st.subheader("📝 Summary")
            if st.button("Summarize Selected", type="secondary", use_container_width=True):
                selected_sources = [st.session_state.research_results[i] for i in st.session_state.selected_snippets if i < len(st.session_state.research_results)]
                if selected_sources:
                    snippet_texts = []
                    for src in selected_sources:
                        filename = src.get('filename', 'Document')
                        snippet = src.get('snippet', src.get('preview', ''))
                        snippet_texts.append(f"From {filename}:\n{snippet}")
                    
                    context = "\n\n---\n\n".join(snippet_texts)
                    summary_messages = [
                        {"role": "system", "content": "You are a helpful assistant that summarizes information from provided document snippets."},
                        {"role": "user", "content": f"Please provide a concise summary:\n\n{context}"}
                    ]
                    
                    with st.spinner("Generating summary..."):
                        try:
                            response = st.session_state.openai_client.get_chat_completion(
                                messages=summary_messages,
                                context=None,
                                stream=False
                            )
                            if hasattr(response, 'choices') and len(response.choices) > 0:
                                summary = response.choices[0].message.content
                                st.markdown(summary)
                        except Exception as e:
                            error_msg = str(e)
                            api_error_logger.error(f"Failed to summarize in research mode: {error_msg}")
                            st.error(f"Failed to summarize: {error_msg}")
    
    with research_col2:
        st.subheader("📚 Search Results")
        
        # Display research results
        if st.session_state.research_results:
            st.success(f"✅ Found {len(st.session_state.research_results)} result(s)")
            st.divider()
            
            # Group by filename
            sources_by_file = {}
            for idx, result in enumerate(st.session_state.research_results):
                filename = result.get('filename', f'Document {idx + 1}')
                if filename not in sources_by_file:
                    sources_by_file[filename] = []
                sources_by_file[filename].append((idx, result))
            
            # Display grouped results with checkboxes
            for filename, sources_list in sources_by_file.items():
                display_name = filename.split('/')[-1] if '/' in filename else filename
                
                with st.expander(f"📄 {display_name} ({len(sources_list)} snippet(s))", expanded=False):
                    for idx, result in sources_list:
                        # Checkbox for selection
                        selected = st.checkbox(
                            f"Select snippet {idx + 1}",
                            value=idx in st.session_state.selected_snippets,
                            key=f"research_snippet_{idx}",
                            label_visibility="visible"
                        )
                        
                        if selected and idx not in st.session_state.selected_snippets:
                            st.session_state.selected_snippets.append(idx)
                        elif not selected and idx in st.session_state.selected_snippets:
                            st.session_state.selected_snippets.remove(idx)
                        
                        # Show exact snippet
                        snippet = result.get('snippet', result.get('preview', result.get('content', '')))
                        if snippet:
                            st.markdown(snippet)
                        
                        # Show relevance score
                        score = result.get('score')
                        if score is not None:
                            st.caption(f"Relevance: {score:.3f}")
                        
                        if idx < sources_list[-1][0]:
                            st.divider()
        elif research_query:
            st.info("No documents found. Try a different search query or adjust filters.")
        else:
            st.info("Enter a search query and click Search to explore documents.")
    
    # Debug Panel (only show if debug mode is enabled)
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
            api_status = {
                "Responses API Available": st.session_state.openai_client.responses_api_available,
                "Last API Used": st.session_state.openai_client.last_api_used or "None",
                "Responses API Attempted": getattr(st.session_state.openai_client, 'responses_api_attempted', False)
            }
            st.json(api_status)
            
            # Show error if Responses API was attempted but failed
            if (st.session_state.openai_client.responses_api_available and 
                st.session_state.openai_client.last_api_used == "fallback_rag" and
                getattr(st.session_state.openai_client, 'last_error', None)):
                st.error(f"**Responses API Error:** {st.session_state.openai_client.last_error}")
                st.caption("The Responses API is available but failed. Falling back to direct vector search + Chat Completions.")
            
            if st.session_state.research_results:
                st.subheader("Latest Research Query Details")
                for idx, result in enumerate(st.session_state.research_results):
                    if '_debug' in result:
                        st.markdown(f"**Result {idx + 1}:**")
                        st.json(result['_debug'])
                        st.divider()


