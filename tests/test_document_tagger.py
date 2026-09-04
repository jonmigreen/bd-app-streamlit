"""Tests for document tagging, attribute coercion, and filter construction.

No network: the OpenAI client is faked. The point of most of these is the
attribute limits -- exceeding 16 keys or 256 characters surfaces as an opaque
400 from the API, so the clamping has to be right here.
"""
import hashlib

import pytest

from conftest import ns
from document_tagger import (
    DOC_TYPES,
    MAX_ATTRIBUTE_KEYS,
    MAX_ATTRIBUTE_VALUE_CHARS,
    OUTCOMES,
    TAG_SCHEMA,
    TOPICS,
    attributes_for,
    build_filter,
    coerce_attributes,
    existing_fingerprints,
    existing_hashes,
    sha256_of,
    tag_document,
)


# ==========================================================================
# Hashing / dedupe
# ==========================================================================

def test_sha256_matches_hashlib(tmp_path):
    payload = b"Civilian RFP response, 2024."
    path = tmp_path / "doc.txt"
    path.write_bytes(payload)

    assert sha256_of(path) == hashlib.sha256(payload).hexdigest()


def test_sha256_is_content_not_name_based(tmp_path):
    (tmp_path / "a.txt").write_bytes(b"same")
    (tmp_path / "b.txt").write_bytes(b"same")

    assert sha256_of(tmp_path / "a.txt") == sha256_of(tmp_path / "b.txt")


def test_sha256_fits_attribute_value_limit(tmp_path):
    path = tmp_path / "doc.txt"
    path.write_bytes(b"x")
    assert len(sha256_of(path)) <= MAX_ATTRIBUTE_VALUE_CHARS


# ==========================================================================
# Attribute coercion -- the API limits
# ==========================================================================

def test_caps_at_sixteen_keys():
    raw = {f"key_{i}": f"value_{i}" for i in range(30)}

    result = coerce_attributes(raw)

    assert len(result) == MAX_ATTRIBUTE_KEYS


def test_truncates_long_values():
    result = coerce_attributes({"client": "C" * 500})

    assert len(result["client"]) == MAX_ATTRIBUTE_VALUE_CHARS


def test_numbers_become_floats():
    """Numeric attributes must be floats so gte/lte range filters work."""
    result = coerce_attributes({"year": 2024})

    assert result["year"] == 2024.0
    assert isinstance(result["year"], float)


def test_booleans_are_preserved_not_stringified():
    result = coerce_attributes({"is_final": True})

    assert result["is_final"] is True


def test_drops_empty_values():
    result = coerce_attributes({"client": "", "agency": None, "sector": "public_health"})

    assert result == {"sector": "public_health"}


def test_drops_year_zero_sentinel():
    """year=0 is the schema's 'not determinable'; storing it would corrupt
    range filters by presenting a real-looking value."""
    result = coerce_attributes({"year": 0, "client": "CDPH"})

    assert "year" not in result
    assert result["client"] == "CDPH"


def test_unsupported_types_are_stringified():
    result = coerce_attributes({"topics": ["a", "b"]})

    assert isinstance(result["topics"], str)


def test_attributes_for_adds_provenance():
    result = attributes_for({"client": "DCC"}, "abc123", "rfp.pdf")

    assert result["client"] == "DCC"
    assert result["content_sha256"] == "abc123"
    assert result["source_filename"] == "rfp.pdf"


# ==========================================================================
# Filter construction
# ==========================================================================

def test_no_filters_returns_none():
    """None means 'send no filter', not 'match nothing'."""
    assert build_filter(None) is None
    assert build_filter({}) is None
    assert build_filter({"client": ""}) is None
    assert build_filter({"client": "All"}) is None


def test_single_filter_is_bare_comparison():
    assert build_filter({"client": "CDPH"}) == {
        "type": "eq", "key": "client", "value": "CDPH"
    }


def test_multiple_filters_wrapped_in_and():
    result = build_filter({"client": "CDPH", "doc_type": "rfp_response"})

    assert result["type"] == "and"
    assert len(result["filters"]) == 2
    assert {"type": "eq", "key": "client", "value": "CDPH"} in result["filters"]


def test_numeric_filter_value_is_float():
    result = build_filter({"year": 2024})

    assert result["value"] == 2024.0
    assert isinstance(result["value"], float)


def test_empty_values_dropped_from_compound():
    result = build_filter({"client": "CDPH", "year": "", "doc_type": "All"})

    assert result == {"type": "eq", "key": "client", "value": "CDPH"}


# ==========================================================================
# Schema
# ==========================================================================

def test_schema_is_strict_compatible():
    """Structured outputs with strict:true require every property listed in
    `required` and additionalProperties disabled."""
    assert TAG_SCHEMA["additionalProperties"] is False
    assert set(TAG_SCHEMA["required"]) == set(TAG_SCHEMA["properties"])


def test_schema_stays_within_key_budget():
    # +2 for content_sha256 and source_filename added at ingest time
    assert len(TAG_SCHEMA["properties"]) + 2 <= MAX_ATTRIBUTE_KEYS


def test_enums_include_fallback_values():
    """Every categorical field needs an escape hatch, or the model is forced
    to guess when the document does not say."""
    assert "other" in DOC_TYPES
    assert "none" in TOPICS
    assert "unknown" in OUTCOMES


# ==========================================================================
# tag_document
# ==========================================================================

class FakeResponses:
    def __init__(self, output_text):
        self.output_text = output_text
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return ns(output_text=self.output_text)


class FakeOpenAI:
    def __init__(self, output_text='{"client": "CDPH", "year": 2024}'):
        self.responses = FakeResponses(output_text)


def test_tag_document_parses_json():
    client = FakeOpenAI()

    tags = tag_document(client, "file_abc")

    assert tags == {"client": "CDPH", "year": 2024}


def test_tag_document_sends_file_as_input_file():
    """The model reads the document directly, so no local PDF parsing."""
    client = FakeOpenAI()

    tag_document(client, "file_abc", filename="rfp.pdf")

    content = client.responses.calls[0]["input"][0]["content"]
    assert content[0] == {"type": "input_file", "file_id": "file_abc"}
    assert "rfp.pdf" in content[1]["text"]


def test_tag_document_requests_strict_structured_output():
    """strict:true is what prevents enum drift breaking eq filters."""
    client = FakeOpenAI()

    tag_document(client, "file_abc")

    fmt = client.responses.calls[0]["text"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["strict"] is True
    assert fmt["schema"] == TAG_SCHEMA


def test_tag_document_raises_on_empty_output():
    client = FakeOpenAI(output_text="")

    with pytest.raises(RuntimeError, match="No tags returned"):
        tag_document(client, "file_abc")


# ==========================================================================
# existing_hashes -- the dedupe manifest
# ==========================================================================

class FakeVectorStoreFiles:
    def __init__(self, files):
        self._files = files

    def list(self, vector_store_id):
        return iter(self._files)


class FakeStoreClient:
    def __init__(self, files):
        self.vector_stores = ns(files=FakeVectorStoreFiles(files))


def test_existing_hashes_maps_digest_to_file_id():
    client = FakeStoreClient([
        ns(id="file_1", attributes={"content_sha256": "aaa"}),
        ns(id="file_2", attributes={"content_sha256": "bbb"}),
    ])

    assert existing_hashes(client, "vs_1") == {"aaa": "file_1", "bbb": "file_2"}


def test_existing_hashes_ignores_untagged_files():
    """Backfill has not run yet -- untagged files must not break ingest."""
    client = FakeStoreClient([
        ns(id="file_1", attributes=None),
        ns(id="file_2", attributes={}),
        ns(id="file_3", attributes={"content_sha256": "ccc"}),
    ])

    assert existing_hashes(client, "vs_1") == {"ccc": "file_3"}


def test_fingerprints_track_filenames_when_hash_is_absent():
    """Files uploaded with purpose='assistants' cannot be downloaded, so the
    backfill cannot hash them -- filename is the only dedupe key they have."""
    client = FakeStoreClient([
        ns(id="file_1", attributes={"source_filename": "sdcp_rfp.docx"}),
        ns(id="file_2", attributes={"content_sha256": "bbb",
                                    "source_filename": "other.pdf"}),
    ])

    hashes, filenames = existing_fingerprints(client, "vs_1")

    assert hashes == {"bbb": "file_2"}
    assert filenames == {"sdcp_rfp.docx": "file_1", "other.pdf": "file_2"}
