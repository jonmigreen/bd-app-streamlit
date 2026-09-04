"""Shared fixtures and fakes for the test suite.

Nothing here touches the network or requires a real API key: the OpenAI SDK
client is replaced with a MagicMock, and Config values are stubbed so tests
pass in CI where no .env exists.
"""
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Make the project root importable regardless of where pytest is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config  # noqa: E402


def ns(**kwargs):
    """Build a simple attribute bag.

    SimpleNamespace, not MagicMock: the response parsers probe with hasattr(),
    and a MagicMock answers True to every hasattr, which would silently defeat
    the branch being tested.
    """
    return types.SimpleNamespace(**kwargs)


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def stub_config(monkeypatch):
    """Pin Config to known values so tests never depend on the real .env."""
    monkeypatch.setattr(Config, "OPENAI_API_KEY", "sk-test-key")
    monkeypatch.setattr(Config, "OPENAI_VECTOR_STORE_ID", "vs_test")
    monkeypatch.setattr(Config, "OPENAI_MODEL", "gpt-5.6-terra")
    monkeypatch.setattr(Config, "OPENAI_REASONING_EFFORT", "low")


# --------------------------------------------------------------------------
# OpenAIClient with a mocked SDK
# --------------------------------------------------------------------------

@pytest.fixture
def client(monkeypatch):
    """An OpenAIClient whose underlying SDK client is a MagicMock."""
    import openai_client as oc

    monkeypatch.setattr(oc, "OpenAI", lambda **kwargs: MagicMock())
    instance = oc.OpenAIClient()

    # Pre-seed the filename cache so _get_filename() never calls
    # client.files.retrieve() and hand back a MagicMock as a "filename".
    instance._file_info_cache.update(
        {f"file_{i}": f"doc_{i}.pdf" for i in range(1, 21)}
    )
    return instance


# --------------------------------------------------------------------------
# Fake OpenAI response objects
# --------------------------------------------------------------------------

def make_annotation(file_id):
    """A flat-shape annotation (direct file_id, no nested file_citation)."""
    return ns(file_id=file_id, start_index=0, end_index=10)


def make_nested_annotation(file_id):
    """A nested-shape annotation (annotation.file_citation.file_id)."""
    return ns(file_citation=ns(file_id=file_id), start_index=0, end_index=10)


def make_file_search_result(file_id, content, score, filename=None):
    return ns(
        file_id=file_id,
        content=content,
        score=score,
        filename=filename or f"{file_id}.pdf",
    )


def make_responses_response(text, cited_file_ids=(), search_results=()):
    """Mimic a Responses API result carrying citations and file_search output."""
    message_item = ns(
        content=[ns(text=text, annotations=[make_annotation(f) for f in cited_file_ids])]
    )
    output = [message_item]
    if search_results:
        # A file_search_call item has no `content`, so the citation extractor
        # skips it and the file_search extractor picks it up.
        output.append(ns(type="file_search_call", results=list(search_results)))
    return ns(output_text=text, output=output)


def make_chat_response(content):
    """Mimic a Chat Completions result."""
    return ns(choices=[ns(message=ns(content=content))])


# --------------------------------------------------------------------------
# Fake client for front-end (AppTest) tests
# --------------------------------------------------------------------------

class FakeClient:
    """Stands in for OpenAIClient in st.session_state.

    Records the arguments it was called with so tests can assert that UI
    controls actually reach the client layer.
    """

    def __init__(self, answer="Fake answer.", sources=None, results=None, error=None,
                 attribute_values=None):
        self.answer = answer
        self.sources = sources if sources is not None else []
        self.results = results if results is not None else []
        self.error = error
        # Empty by default, which hides the filter UI -- matching an untagged
        # corpus, the state the app is in before a backfill runs.
        self.attribute_values = attribute_values or {}

        # Attributes app.py reads directly
        self.responses_api_available = True
        self.last_api_used = "responses_api"
        self.last_error = None
        self.last_filtered_count = 0
        self.last_threshold_applied = None

        # Call recording
        self.rag_calls = []
        self.search_calls = []
        self.chunk_calls = []

    def get_rag_response(self, user_query, conversation_history,
                         min_relevance_score=None, filters=None):
        self.rag_calls.append({
            "query": user_query,
            "min_relevance_score": min_relevance_score,
            "filters": filters,
        })
        if self.error:
            raise self.error
        self.last_threshold_applied = min_relevance_score or 0.0
        return self.answer, self.sources

    def search_vectors(self, query, top_k=50, min_relevance_score=None, filters=None):
        self.search_calls.append({
            "query": query,
            "top_k": top_k,
            "min_relevance_score": min_relevance_score,
            "filters": filters,
        })
        if self.error:
            raise self.error
        return self.results

    def list_attribute_values(self, keys):
        return {k: v for k, v in self.attribute_values.items() if k in keys}

    def ask_about_chunks(self, question, chunks):
        self.chunk_calls.append({"question": question, "chunks": chunks})
        if self.error:
            raise self.error
        return self.answer, chunks


def make_snippet(idx, filename="doc.pdf", score=0.9):
    """A search-result dict shaped the way research_page() expects."""
    return {
        "filename": filename,
        "snippet": f"Snippet body {idx}",
        "preview": f"Preview {idx}",
        "content": f"Snippet body {idx}",
        "score": score,
        "file_id": f"file_{idx}",
        "metadata": {"filename": filename},
    }
