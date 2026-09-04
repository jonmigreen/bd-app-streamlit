"""Main Streamlit application for OpenAI chat with vector store RAG."""

import json
from datetime import datetime
from typing import List, Dict

import streamlit as st

from openai_client import OpenAIClient
from config import Config
from logger import api_error_logger

from utility import check_password
# Page configuration
st.set_page_config(
    page_title="OpenAI Chat with RAG",
    page_icon="💬",
    layout="wide"
)

# Do not continue if check_password is not True.  
if not check_password():  
    st.stop()
# Initialize session state
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "messages": [],
        "message_sources": {},  # Map message index to sources
        "debug_mode": False,
        "research_results": [],
        "relevance_threshold": 0.55,
        "selected_snippets": [],
        # Bumped on each search so snippet checkboxes get fresh widget keys
        # and cannot carry selections over to an unrelated result set.
        "search_generation": 0,
        "filters": {},
        "current_sources": [],
        "research_answer": None,  # Research mode answer
        "research_cited_sources": [],  # Research mode cited sources
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


FILTER_KEYS = ("client", "year", "doc_type", "outcome")
FILTER_LABELS = {
    "client": "🏢 Client",
    "year": "📅 Year",
    "doc_type": "📄 Document Type",
    "outcome": "🏆 Outcome",
}


@st.cache_data(ttl=300, show_spinner=False)
def load_filter_options():
    """Distinct tag values across the corpus, for the filter dropdowns.

    Cached because it lists every file in the store. Returns {} when nothing
    is tagged yet, which hides the filter UI entirely.
    """
    return st.session_state.openai_client.list_attribute_values(FILTER_KEYS)


def render_filters():
    """Filter controls, shown only when the corpus actually has tags."""
    options = load_filter_options()
    if not options:
        return

    with st.expander("🔎 Filters", expanded=any(st.session_state.filters.values())):
        selected = {}
        for key in FILTER_KEYS:
            values = options.get(key)
            # A dropdown offering a single value cannot narrow anything --
            # e.g. every document is an rfp_response. Hide it as noise.
            if not values or len(values) < 2:
                continue
            # Year is stored as a float; render it as a plain year.
            display = ["All"] + [
                str(int(v)) if key == "year" and isinstance(v, float) else str(v)
                for v in values
            ]
            choice = st.selectbox(
                FILTER_LABELS[key], display, key=f"filter_{key}"
            )
            if choice != "All":
                selected[key] = float(choice) if key == "year" else choice

        st.session_state.filters = selected

        if selected:
            st.caption(f"Filtering on {len(selected)} attribute(s)")
            if st.button("Clear filters", use_container_width=True):
                for key in FILTER_KEYS:
                    st.session_state.pop(f"filter_{key}", None)
                st.session_state.filters = {}
                st.rerun()


def render_sidebar():
    """Render sidebar settings that appear on all pages."""
    with st.sidebar:
        render_filters()

        # Settings
        with st.expander("⚙️ Settings", expanded=False):
            st.info(f"**Model:** {Config.OPENAI_MODEL}")

            if st.session_state.debug_mode:
                vs_id = Config.OPENAI_VECTOR_STORE_ID or "Not configured"
                st.info(f"**Vector Store ID:** `{vs_id}`")

                client = st.session_state.openai_client
                api_status = (
                    "✅ Responses API"
                    if client.responses_api_available
                    else "❌ Direct Search Only"
                )
                st.info(f"**API Mode:** {api_status}")

                if client.last_api_used:
                    st.caption(f"**Last Used:** `{client.last_api_used}`")
            else:
                st.info("Vector store configured ✓")

            st.divider()

            st.session_state.debug_mode = st.checkbox(
                "🐛 Debug Mode",
                value=st.session_state.debug_mode,
                help="Show technical details",
            )

            st.divider()

            st.session_state.relevance_threshold = st.slider(
                "📊 Min Relevance Score",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.relevance_threshold,
                step=0.05,
                help="Filter sources below this threshold",
            )

            st.divider()

            if st.button(
                "🗑️ Clear Chat History", type="secondary", use_container_width=True
            ):
                st.session_state.messages = []
                st.session_state.message_sources = {}
                st.session_state.research_results = []
                st.session_state.current_sources = []
                st.session_state.selected_snippets = []
                st.rerun()


TAG_DISPLAY_KEYS = ("client", "year", "doc_type", "sector", "primary_topic", "outcome")


def format_tags(metadata: Dict) -> str:
    """Render a source's tags as a compact caption line."""
    parts = []
    for key in TAG_DISPLAY_KEYS:
        value = metadata.get(key)
        if value is None or value in ("", "unknown", "none", "other"):
            continue
        if key == "year" and isinstance(value, float):
            value = int(value)
        parts.append(f"{key.replace('_', ' ')}: **{value}**")
    return " · ".join(parts)


def display_source_expander(sources: List[Dict], title_prefix: str = ""):
    """Display sources with expandable full passages."""
    if not sources:
        return

    with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
        for num, source in enumerate(sources, 1):
            filename = source.get("filename", f"Document {num}")
            display_name = filename.split("/")[-1] if "/" in filename else filename

            st.markdown(f"**{num}. {display_name}**")

            # Tags, so users can see why a document matched
            tags = format_tags(source.get("metadata") or {})
            if tags:
                st.caption(tags)

            # Expandable full passage (from file_search results)
            full_content = source.get("content", "")
            if full_content:
                with st.expander("📖 Expand to see full passage", expanded=False):
                    st.markdown(full_content)

            # Show relevance score if available
            score = source.get("score")
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
                {"filename": src.get("filename", "Unknown"), "score": src.get("score")}
                for src in sources[:5]
            ],
            "filters": filters,
        }

        with open("query_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception:
        pass  # Don't fail if logging fails


# Page functions
def about_page():
    """About page content."""
    render_sidebar()
    st.title("About")
    st.markdown("""
    
    This application provides intelligent access to Civilian's business development documents. It searches our documents, pulls relevant snippets, and uses them as context for AI generated answers with citations.
    
    ---
    ## ⭐ How it Works

    ### 💬 Chat

    Ask questions in natural language and get AI-generated answers using relevant BD documents as context.

    ### 🔍 Research Mode

    Search the database directly using keywords, phrases, or sentences. Relevant snippets are returned. 
    You can then select snippets to use as context for a follow-up question.
  
    ---

    ## Example Questions:

        - How does Civilian approach behavior-change based on marketing? 
        - Pull three examples where we describe behavior change models.  
        - How do we talk about co-creation with community-based organizations?
        - Summarize common challenges we cite in our proposals.
        - Summarize Civilian's approach to campaign development.
        - Describe Civilian's approach to paid media management.
        - Find all references to subcontractor management across the dataset and summarize the specific process Civilian utilizes.
        - Retrieve and cite each RFP and page number where we discuss our co-creation process in detail.
        - Find, summarize, and cite all references to the use of translation and transadaptation services.
    ---
    ## Data & Limitations
    
    - The BD Assistant can only use documents that have been loaded into the database. It currently contains ~20 RFP response documents.  
    - If an answer does **not** include citations, treat it as a suggestion and verify manually.  
    - Some documents may have limited or imperfect text extraction.  
    - This is a **pilot**: quality will improve as we add more documents, refine prompts, and adjust search settings.
    ---

    ## Feedback
    This is an early pilot. Please tell us:
    - Where answers were especially helpful
    - Where answers were wrong, incomplete, or confusing
    - Gaps you noticed in the document set (e.g., “we need the final X RFP in here”)
    ---
    """
    )


def chat_page():
    """Chat page content."""
    render_sidebar()
    st.title("Chat with BD Knowledge Base")
    st.markdown(
        """
        **Ask questions and get answers enhanced with context from your vector store.**
        1. You ask a question.
        2. The system searches our BD document library and pulls the most relevant passages.
        3. The AI reads those passages and drafts an answer using that content.
        4. It shows the answer along with citations and snippets so you can verify and reuse the language.
        ---
        """
    )

    # Display chat history
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Display sources for assistant messages
            if (
                message["role"] == "assistant"
                and idx in st.session_state.message_sources
            ):
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

                    response_text, sources = (
                        st.session_state.openai_client.get_rag_response(
                            user_query=prompt,
                            conversation_history=conversation_history,
                            min_relevance_score=st.session_state.relevance_threshold,
                            filters=st.session_state.filters,
                        )
                    )

                    # Debug info
                    client = st.session_state.openai_client
                    if st.session_state.debug_mode:
                        if client.last_threshold_applied is not None:
                            # file_search filters server-side, so it never
                            # reports how many chunks it dropped.
                            st.info(
                                f"🔍 {len(sources)} source(s) returned at or above "
                                f"relevance {client.last_threshold_applied:.2f}"
                            )
                        elif client.last_filtered_count > 0:
                            st.info(
                                f"🔍 Filtered {client.last_filtered_count} source(s) below threshold"
                            )

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
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response_text}
                    )

                except ValueError as e:
                    error_msg = f"⚠️ Configuration Error: {str(e)}"
                    api_error_logger.error(f"Config error: {str(e)}")
                    message_placeholder.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )

                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    api_error_logger.error(f"API error: {str(e)}")
                    message_placeholder.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )


def research_page():
    """Research Mode page content."""
    render_sidebar()
    st.title("Research Mode")
    st.markdown("""
        **Search documents, select relevant snippets, and ask questions about them.**
        1. Enter a search query to find relevant document chunks.
        2. Review the results and select the snippets you want to work with.
        3. Ask a follow-up question about your selected snippets.
        4. The AI Assistant will generate an answer using **only** those selected snippets and return citations.
        """
    )
    st.divider()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🔍 Search")

        # Search
        query = st.text_input("Search query:", placeholder="e.g., fiscal management")

        if st.button("🔍 Search", type="primary", use_container_width=True):
            if query:
                with st.spinner("Searching..."):
                    try:
                        st.session_state.research_results = st.session_state.openai_client.search_vectors(
                            query=query,
                            top_k=50,
                            min_relevance_score=st.session_state.relevance_threshold,
                            filters=st.session_state.filters,
                        )
                        st.session_state.selected_snippets = []
                        st.session_state.search_generation += 1
                        st.session_state.research_answer = None
                        st.session_state.research_cited_sources = []
                    except Exception as e:
                        api_error_logger.error(f"Search failed: {str(e)}")
                        st.error(f"Search failed: {str(e)}")
                        st.session_state.research_results = []
            else:
                st.warning("Please enter a search query")

        # Research Mode: Ask about selected chunks
        if st.session_state.selected_snippets and st.session_state.research_results:
            st.divider()
            st.subheader("💡 Research Mode")
            st.caption(f"{len(st.session_state.selected_snippets)} snippet(s) selected")

            # Question input
            research_question = st.text_area(
                "Ask a question about the selected snippets:",
                placeholder="e.g., What are the key themes across these documents?",
                height=100,
                key="research_question",
            )

            if st.button(
                "🔎 Analyze Selected", type="primary", use_container_width=True
            ):
                if research_question.strip():
                    selected = [
                        st.session_state.research_results[i]
                        for i in st.session_state.selected_snippets
                        if i < len(st.session_state.research_results)
                    ]

                    if selected:
                        with st.spinner("Analyzing selected snippets..."):
                            try:
                                answer, cited_sources = (
                                    st.session_state.openai_client.ask_about_chunks(
                                        question=research_question, chunks=selected
                                    )
                                )

                                # Store results in session state for display
                                st.session_state.research_answer = answer
                                st.session_state.research_cited_sources = cited_sources

                            except Exception as e:
                                api_error_logger.error(
                                    f"Research query failed: {str(e)}"
                                )
                                st.error(f"Failed to analyze: {str(e)}")
                else:
                    st.warning("Please enter a question")

            # Display research answer if available
            if (
                "research_answer" in st.session_state
                and st.session_state.research_answer
            ):
                st.divider()
                st.markdown("### Answer")
                st.markdown(st.session_state.research_answer)

                # Show cited sources
                if (
                    "research_cited_sources" in st.session_state
                    and st.session_state.research_cited_sources
                ):
                    with st.expander(
                        f"📚 Sources Referenced ({len(st.session_state.research_cited_sources)})",
                        expanded=False,
                    ):
                        for i, src in enumerate(
                            st.session_state.research_cited_sources, 1
                        ):
                            filename = src.get("filename", f"Document {i}")
                            display_name = (
                                filename.split("/")[-1] if "/" in filename else filename
                            )
                            score = src.get("score")

                            st.markdown(f"**{i}. {display_name}**")
                            if score is not None:
                                st.caption(f"Relevance: {score:.3f}")

                            if i < len(st.session_state.research_cited_sources):
                                st.divider()

    with col2:
        st.subheader("📚 Search Results")

        results = st.session_state.research_results

        if results:
            st.success(f"✅ Found {len(results)} result(s)")
            st.divider()

            # Group by filename
            by_file = {}
            for idx, result in enumerate(results):
                filename = result.get("filename", f"Document {idx + 1}")
                if filename not in by_file:
                    by_file[filename] = []
                by_file[filename].append((idx, result))

            # Display grouped results
            for filename, items in by_file.items():
                display_name = filename.split("/")[-1] if "/" in filename else filename

                with st.expander(
                    f"📄 {display_name} ({len(items)} snippet(s))", expanded=False
                ):
                    for idx, result in items:
                        # Selection checkbox
                        # Keyed per search generation and given no `value=`:
                        # the widget's own state is the single source of truth,
                        # so a new search starts from unchecked boxes.
                        selected = st.checkbox(
                            f"Select snippet {idx + 1}",
                            key=f"snippet_{st.session_state.search_generation}_{idx}",
                        )

                        if selected and idx not in st.session_state.selected_snippets:
                            st.session_state.selected_snippets.append(idx)
                        elif not selected and idx in st.session_state.selected_snippets:
                            st.session_state.selected_snippets.remove(idx)

                        # Display snippet
                        snippet = result.get("snippet") or result.get("content", "")
                        if snippet:
                            st.markdown(snippet)

                        # Show score
                        score = result.get("score")
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
            st.json(
                {
                    "total_messages": len(st.session_state.messages),
                    "messages_with_sources": len(st.session_state.message_sources),
                    "research_results_count": len(st.session_state.research_results),
                    "vector_store_id": Config.OPENAI_VECTOR_STORE_ID,
                    "model": Config.OPENAI_MODEL,
                }
            )

            st.subheader("API Usage")
            client = st.session_state.openai_client
            st.json(
                {
                    "Responses API Available": client.responses_api_available,
                    "Last API Used": client.last_api_used or "None",
                    "Last Error": client.last_error or "None",
                }
            )


# Navigation setup
def main():
    """Build navigation and run the selected page."""
    pages = [
        st.Page(about_page, title="About", icon="💡"),
        st.Page(chat_page, title="Chat", icon="💬"),
        st.Page(research_page, title="Research Mode", icon="🔍"),
    ]
    st.navigation(pages).run()


# Streamlit runs the main script with __name__ == "__main__", so the app
# behaves identically. Importing this module (as the tests do) skips the
# nav run, letting each page function be rendered in isolation.
if __name__ == "__main__":
    main()
