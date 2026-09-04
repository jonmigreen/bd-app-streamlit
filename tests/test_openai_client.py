"""Tests for OpenAIClient: API contract, citation parsing, relevance filtering.

openai_client.py imports no Streamlit, so it is directly importable here.
The SDK client is a MagicMock (see conftest.client), so nothing hits the network.
"""
import pytest

from conftest import (
    make_chat_response,
    make_file_search_result,
    make_nested_annotation,
    make_responses_response,
    ns,
)


# ==========================================================================
# Migration regressions: the parameters actually sent to OpenAI
# ==========================================================================

def test_responses_call_sends_score_threshold(client):
    """Bug 4.1: the sidebar relevance slider must reach file_search.

    On main the threshold was accepted and silently dropped, making the
    slider a no-op in Chat mode.
    """
    client.client.responses.create.return_value = make_responses_response("Answer.")

    client._get_rag_response_via_responses_api("a question", 0.72)

    kwargs = client.client.responses.create.call_args.kwargs
    ranking = kwargs["tools"][0]["ranking_options"]
    assert ranking["score_threshold"] == 0.72
    assert ranking["ranker"] == "auto"


def test_threshold_threads_all_the_way_from_get_rag_response(client):
    """The public entry point must forward the threshold, not drop it."""
    client.client.responses.create.return_value = make_responses_response("Answer.")

    client.get_rag_response("q", conversation_history=[], min_relevance_score=0.81)

    kwargs = client.client.responses.create.call_args.kwargs
    assert kwargs["tools"][0]["ranking_options"]["score_threshold"] == 0.81
    assert client.last_threshold_applied == 0.81


def test_absent_threshold_defaults_to_zero_not_none(client):
    """score_threshold must be a float; None would be rejected by the API."""
    client.client.responses.create.return_value = make_responses_response("Answer.")

    client._get_rag_response_via_responses_api("q", None)

    kwargs = client.client.responses.create.call_args.kwargs
    assert kwargs["tools"][0]["ranking_options"]["score_threshold"] == 0.0


def test_responses_call_sends_nested_reasoning(client):
    """Responses API takes reasoning={"effort": ...} (nested)."""
    client.client.responses.create.return_value = make_responses_response("Answer.")

    client._get_rag_response_via_responses_api("q", 0.5)

    kwargs = client.client.responses.create.call_args.kwargs
    assert kwargs["reasoning"] == {"effort": "low"}
    assert kwargs["include"] == ["file_search_call.results"]
    assert kwargs["model"] == "gpt-5.6-terra"


def test_responses_call_sends_no_temperature(client):
    """GPT-5.x rejects temperature unless effort is 'none'."""
    client.client.responses.create.return_value = make_responses_response("Answer.")

    client._get_rag_response_via_responses_api("q", 0.5)

    assert "temperature" not in client.client.responses.create.call_args.kwargs


def test_chat_completion_sends_flat_reasoning_effort(client):
    """Chat Completions takes reasoning_effort=... (flat, not nested).

    Getting these two spellings backwards is the classic migration error.
    """
    client.client.chat.completions.create.return_value = make_chat_response("Hi.")

    client.get_chat_completion([{"role": "user", "content": "hi"}])

    kwargs = client.client.chat.completions.create.call_args.kwargs
    assert kwargs["reasoning_effort"] == "low"
    assert "temperature" not in kwargs
    assert "reasoning" not in kwargs


def test_ask_about_chunks_sends_flat_reasoning_effort(client):
    client.client.chat.completions.create.return_value = make_chat_response("Hi.")

    client.ask_about_chunks("q", [{"filename": "a.pdf", "content": "text"}])

    kwargs = client.client.chat.completions.create.call_args.kwargs
    assert kwargs["reasoning_effort"] == "low"
    assert "temperature" not in kwargs


# ==========================================================================
# Bug 4.2: citation attribution must not match substrings
# ==========================================================================

def _chunks(n):
    return [{"filename": f"doc_{i}.pdf", "content": f"body {i}"} for i in range(1, n + 1)]


def test_citation_12_does_not_also_match_source_1(client):
    """Bug 4.2: 'Source 1' is a substring of 'Source 10'..'Source 19'.

    Fails on main, where chunk 1 is wrongly reported as cited.
    """
    client.client.chat.completions.create.return_value = make_chat_response(
        "The process is described in [Source 12]."
    )

    _, cited = client.ask_about_chunks("q", _chunks(12))

    assert len(cited) == 1
    assert cited[0]["filename"] == "doc_12.pdf"


def test_multiple_citations_all_resolved(client):
    client.client.chat.completions.create.return_value = make_chat_response(
        "See [Source 2] and [Source 11], but not the others."
    )

    _, cited = client.ask_about_chunks("q", _chunks(12))

    assert [c["filename"] for c in cited] == ["doc_2.pdf", "doc_11.pdf"]


def test_no_citations_falls_back_to_all_chunks(client):
    """Preserved behaviour: an uncited answer still shows its inputs."""
    client.client.chat.completions.create.return_value = make_chat_response(
        "I cannot answer from the provided sources."
    )
    chunks = _chunks(5)

    _, cited = client.ask_about_chunks("q", chunks)

    assert cited == chunks


def test_none_answer_does_not_crash(client):
    """message.content can be None; the `answer or ''` guard must hold."""
    client.client.chat.completions.create.return_value = make_chat_response(None)
    chunks = _chunks(3)

    answer, cited = client.ask_about_chunks("q", chunks)

    assert answer is None
    assert cited == chunks


def test_citation_out_of_range_is_ignored(client):
    """A hallucinated [Source 99] must not raise or invent a source."""
    client.client.chat.completions.create.return_value = make_chat_response(
        "As shown in [Source 99]."
    )
    chunks = _chunks(3)

    _, cited = client.ask_about_chunks("q", chunks)

    assert cited == chunks  # no valid citations -> fallback


def test_empty_chunks_rejected(client):
    with pytest.raises(ValueError, match="No chunks provided"):
        client.ask_about_chunks("q", [])


# ==========================================================================
# Relevance filtering (Research path, unchanged by the migration)
# ==========================================================================

def test_filter_keeps_scores_at_or_above_threshold(client):
    sources = [{"score": 0.9}, {"score": 0.4}, {"score": 0.5}]

    kept, dropped = client._filter_by_relevance(sources, 0.5)

    assert [s["score"] for s in kept] == [0.9, 0.5]
    assert dropped == 1


def test_filter_keeps_unscored_sources(client):
    """score=None means 'unknown', not 'irrelevant'."""
    sources = [{"score": None}, {"score": 0.1}]

    kept, dropped = client._filter_by_relevance(sources, 0.5)

    assert kept == [{"score": None}]
    assert dropped == 1


# ==========================================================================
# Response parsing
# ==========================================================================

def test_clean_citation_markers_strips_openai_markers(client):
    text = "Civilian uses co-creation【4:0†source】 widely."

    cleaned, _ = client._clean_citation_markers(text, [])

    assert "【" not in cleaned
    assert "Civilian uses co-creation widely." == cleaned


def test_parse_annotation_flat_shape(client):
    result = client._parse_annotation(ns(file_id="file_1", start_index=0, end_index=5))
    assert result["file_id"] == "file_1"
    assert result["filename"] == "doc_1.pdf"


def test_parse_annotation_nested_shape(client):
    result = client._parse_annotation(make_nested_annotation("file_2"))
    assert result["file_id"] == "file_2"
    assert result["filename"] == "doc_2.pdf"


def test_parse_annotation_unrecognised_shape_returns_none(client):
    assert client._parse_annotation(ns(irrelevant=True)) is None


def test_extract_file_search_results_string_content(client):
    response = make_responses_response(
        "Answer.",
        search_results=[make_file_search_result("file_1", "Body text", 0.88)],
    )

    results = client._extract_file_search_results(response)

    assert results["file_1"]["content"] == "Body text"
    assert results["file_1"]["score"] == 0.88


def test_extract_file_search_results_block_list_content(client):
    """Content also arrives as a list of blocks; both shapes must work."""
    blocked = ns(
        file_id="file_3",
        content=[ns(text="Part one"), ns(text="Part two")],
        score=0.7,
        filename="doc_3.pdf",
    )
    response = make_responses_response("Answer.", search_results=[blocked])

    results = client._extract_file_search_results(response)

    assert results["file_3"]["content"] == "Part one\n\nPart two"


def test_rag_response_builds_sources_from_citations(client):
    """End-to-end on the Responses path: citations + search results merge."""
    client.client.responses.create.return_value = make_responses_response(
        "Grounded answer.",
        cited_file_ids=["file_1"],
        search_results=[make_file_search_result("file_1", "Full passage", 0.91)],
    )

    text, sources = client._get_rag_response_via_responses_api("q", 0.5)

    assert text == "Grounded answer."
    assert len(sources) == 1
    assert sources[0]["filename"] == "doc_1.pdf"
    assert sources[0]["content"] == "Full passage"
    assert sources[0]["score"] == 0.91
