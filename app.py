"""Main Streamlit application for OpenAI chat with vector store RAG."""
import os
from typing import List, Dict

import streamlit as st
from openai_client import OpenAIClient
from config import Config

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

# App title and description
if st.session_state.research_mode:
    st.title("🔍 Research Mode - Document Explorer")
    st.markdown("Search and explore documents in your vector store.")
else:
    st.title("Business Development Resource Library")
    st.markdown("Ask questions and get answers enhanced with context from your vector store.")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    st.info(f"**Model:** {Config.OPENAI_MODEL}")
    
    # Only show vector store ID in debug mode
    if st.session_state.debug_mode:
        vs_id_display = Config.OPENAI_VECTOR_STORE_ID if Config.OPENAI_VECTOR_STORE_ID else "Not configured"
        st.info(f"**Vector Store ID:** `{vs_id_display}`")
    else:
        st.info("Vector store configured ✓")
    
    st.divider()
    
    # Research Mode toggle
    st.session_state.research_mode = st.checkbox(
        "🔍 Research Mode", 
        value=st.session_state.research_mode,
        help="Explore documents directly without chat"
    )
    
    # Debug Mode toggle (hidden by default, opt-in)
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

# Research Mode UI (Search-first mode)
if st.session_state.research_mode:
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
                        st.error(f"Search failed: {str(e)}")
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
                            st.error(f"Failed to summarize: {str(e)}")
    
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
                
                with st.expander(f"📄 {display_name} ({len(sources_list)} snippet(s))", expanded=True):
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
    
    st.divider()
    st.markdown("---")
    st.markdown("### Chat History")
    
    # Debug Panel (only show in research mode or if debug mode is enabled)
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
            
            if st.session_state.research_results:
                st.subheader("Latest Research Query Details")
                for idx, result in enumerate(st.session_state.research_results):
                    if '_debug' in result:
                        st.markdown(f"**Result {idx + 1}:**")
                        st.json(result['_debug'])
                        st.divider()

# Two-column layout for chat mode
if not st.session_state.research_mode:
    col1, col2 = st.columns([2, 1])
    # col1 and col2 are both defined above and used in separate with blocks below
    
    with col1:
        # Left column: Chat interface
        st.subheader("")
        
        # Display chat history
        for idx, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
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
                
                with st.spinner("Querying vector store and generating response..."):
                    try:
                        # Get conversation history (excluding the current user message for the API call)
                        conversation_history = st.session_state.messages[:-1]
                        
                        # Parallel calls: Assistants API + Vector search
                        # Call 1: Get RAG-enhanced response with sources
                        response_text, sources = st.session_state.openai_client.get_rag_response(
                            user_query=prompt,
                            conversation_history=conversation_history,
                            min_relevance_score=st.session_state.relevance_threshold
                        )
                        
                        # Debug logging for filtered sources in RAG response
                        if st.session_state.debug_mode and st.session_state.openai_client.last_filtered_count > 0:
                            st.info(f"🔍 Filtered out {st.session_state.openai_client.last_filtered_count} source(s) below relevance threshold ({st.session_state.relevance_threshold:.2f}) in RAG response")
                        
                        # Call 2: Get detailed sources using direct vector search (in parallel)
                        search_results = []
                        try:
                            search_results = st.session_state.openai_client.search_vectors(
                                query=prompt,
                                top_k=50,
                                filters=st.session_state.filters if any(st.session_state.filters.values()) else None,
                                min_relevance_score=st.session_state.relevance_threshold
                            )
                            if search_results:
                                st.session_state.current_sources = search_results
                                # Debug logging for filtered sources
                                if st.session_state.debug_mode and search_results and '_debug' in search_results[0]:
                                    filtered_count = search_results[0].get('_debug', {}).get('filtered_count', 0)
                                    if filtered_count > 0:
                                        st.info(f"🔍 Filtered out {filtered_count} source(s) below relevance threshold ({st.session_state.relevance_threshold:.2f})")
                            else:
                                # Empty results - try fallback
                                if st.session_state.debug_mode:
                                    st.warning("Direct vector search returned empty results, trying fallback...")
                                try:
                                    fallback_sources = st.session_state.openai_client.get_sources_for_query(
                                        prompt, max_results=50, min_relevance_score=st.session_state.relevance_threshold
                                    )
                                    if fallback_sources:
                                        st.session_state.current_sources = fallback_sources
                                    else:
                                        st.session_state.current_sources = []
                                        if st.session_state.debug_mode:
                                            st.warning("Fallback search also returned empty results")
                                except Exception as fallback_error:
                                    st.session_state.current_sources = []
                                    if st.session_state.debug_mode:
                                        st.error(f"Fallback search failed: {str(fallback_error)}")
                        except Exception as e:
                            # If direct search fails, try fallback or show error in debug mode
                            error_msg = str(e)
                            if st.session_state.debug_mode:
                                st.warning(f"Direct vector search failed: {error_msg}")
                            # Try fallback: use get_sources_for_query which might work
                            try:
                                fallback_sources = st.session_state.openai_client.get_sources_for_query(
                                    prompt, max_results=50, min_relevance_score=st.session_state.relevance_threshold
                                )
                                if fallback_sources:
                                    st.session_state.current_sources = fallback_sources
                                else:
                                    st.session_state.current_sources = []
                                    if st.session_state.debug_mode:
                                        st.warning("Fallback search returned empty results")
                            except Exception as fallback_error:
                                st.session_state.current_sources = []
                                if st.session_state.debug_mode:
                                    st.error(f"Fallback search also failed: {str(fallback_error)}")
                        
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
                                    for src in st.session_state.current_sources[:5]
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
                        
                        # Display response
                        message_placeholder.markdown(response_text)
                        
                        # Store sources with message index
                        message_idx = len(st.session_state.messages)
                        st.session_state.message_sources[message_idx] = st.session_state.current_sources
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response_text})
                        
                    except ValueError as e:
                        # Configuration errors
                        error_message = f"⚠️ Configuration Error: {str(e)}\n\nPlease check your .env file and ensure all required variables are set."
                        message_placeholder.error(error_message)
                        st.session_state.messages.append({"role": "assistant", "content": error_message})
                        
                    except Exception as e:
                        # API or other errors
                        error_message = f"❌ Error: {str(e)}\n\nPlease try again or check your API configuration."
                        message_placeholder.error(error_message)
                        st.session_state.messages.append({"role": "assistant", "content": error_message})
    
    with col2:
        # Right column: Sources / Research panel
        st.subheader("📚 Sources")
        
        # Filter controls
        with st.expander("🔍 Filters", expanded=False):
            st.session_state.filters["client"] = st.text_input("Client", value=st.session_state.filters.get("client") or "")
            st.session_state.filters["year"] = st.text_input("Year", value=st.session_state.filters.get("year") or "")
            st.session_state.filters["topic"] = st.text_input("Topic", value=st.session_state.filters.get("topic") or "")
        
        # Display current sources
        if st.session_state.current_sources and len(st.session_state.current_sources) > 0:
            st.markdown(f"**Found {len(st.session_state.current_sources)} source(s)**")
            
            # Debug: show source count and first source info if debug mode
            if st.session_state.debug_mode:
                st.caption(f"Debug: {len(st.session_state.current_sources)} sources loaded")
                if st.session_state.current_sources:
                    first_source = st.session_state.current_sources[0]
                    st.caption(f"First source keys: {list(first_source.keys())}")
                    st.caption(f"First source snippet length: {len(first_source.get('snippet', ''))}")
            
            # Group sources by filename
            sources_by_file = {}
            for idx, source in enumerate(st.session_state.current_sources):
                filename = source.get('filename', f'Document {idx + 1}')
                if filename not in sources_by_file:
                    sources_by_file[filename] = []
                sources_by_file[filename].append((idx, source))
            
            # Display grouped sources
            for filename, sources_list in sources_by_file.items():
                display_name = filename.split('/')[-1] if '/' in filename else filename
                
                with st.expander(f"📄 {display_name} ({len(sources_list)} snippet(s))", expanded=False):
                    for idx, source in sources_list:
                        # Checkbox for selection
                        selected = st.checkbox(
                            f"Snippet {idx + 1}",
                            value=idx in st.session_state.selected_snippets,
                            key=f"snippet_{idx}",
                            label_visibility="collapsed"
                        )
                        
                        if selected and idx not in st.session_state.selected_snippets:
                            st.session_state.selected_snippets.append(idx)
                        elif not selected and idx in st.session_state.selected_snippets:
                            st.session_state.selected_snippets.remove(idx)
                        
                        # Show exact snippet
                        snippet = source.get('snippet', source.get('preview', ''))
                        if snippet:
                            st.markdown(snippet)
                        
                        # Show relevance score (always show in sources panel for trust)
                        score = source.get('score')
                        if score is not None:
                            st.caption(f"Relevance: {score:.3f}")
                        
                        if idx < sources_list[-1][0]:
                            st.divider()
            
            # Summarize selected snippets button
            if st.session_state.selected_snippets:
                if st.button("📝 Summarize Selected Snippets", type="primary", use_container_width=True):
                    # Get selected snippets
                    selected_sources = [st.session_state.current_sources[i] for i in st.session_state.selected_snippets if i < len(st.session_state.current_sources)]
                    
                    if selected_sources:
                        # Prepare context from selected snippets
                        snippet_texts = []
                        for src in selected_sources:
                            filename = src.get('filename', 'Document')
                            snippet = src.get('snippet', '')
                            snippet_texts.append(f"From {filename}:\n{snippet}")
                        
                        context = "\n\n---\n\n".join(snippet_texts)
                        
                        # Call Assistants API to summarize
                        with st.spinner("Summarizing selected snippets..."):
                            try:
                                summary_messages = [
                                    {
                                        "role": "system",
                                        "content": "You are a helpful assistant that summarizes information from provided document snippets."
                                    },
                                    {
                                        "role": "user",
                                        "content": f"Please provide a concise summary of the following information:\n\n{context}"
                                    }
                                ]
                                
                                response = st.session_state.openai_client.get_chat_completion(
                                    messages=summary_messages,
                                    context=None,
                                    stream=False
                                )
                                
                                if hasattr(response, 'choices') and len(response.choices) > 0:
                                    summary = response.choices[0].message.content
                                    
                                    # Display summary in left column (add to messages)
                                    st.session_state.messages.append({
                                        "role": "user",
                                        "content": "[Summarize selected snippets from sources]"
                                    })
                                    st.session_state.messages.append({
                                        "role": "assistant",
                                        "content": summary
                                    })
                                    st.rerun()
                                    
                            except Exception as e:
                                st.error(f"Failed to summarize: {str(e)}")
        else:
            # Show helpful message based on state
            if len(st.session_state.messages) > 0:
                # User has asked questions but no sources
                if st.session_state.debug_mode:
                    st.warning("No sources available. Vector search may have failed. Check error messages above.")
                else:
                    st.info("No sources found for the last question. Try asking another question.")
            else:
                st.info("Ask a question to see sources here.")

