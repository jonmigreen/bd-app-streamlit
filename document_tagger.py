"""Document tagging for vector store ingestion.

Tags are stored as OpenAI vector store file *attributes*, which the search API
can filter on server-side. The API constrains them hard: at most 16 keys, and
values must be str, float, or bool -- no lists, no nested objects. That is why
topics are two single-valued keys drawn from a closed taxonomy rather than a
list, and why every categorical field is an enum: free text would make `eq`
filters unreliable (the model writes "Public Health" one day, "public health"
the next, and the filter silently misses).
"""
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

# OpenAI vector store attribute limits
MAX_ATTRIBUTE_KEYS = 16
MAX_ATTRIBUTE_VALUE_CHARS = 256

# Model used for tagging. Extracting a client name and a year is not a
# reasoning task, so the cheapest current-generation model is the right one.
TAGGING_MODEL = "gpt-5.6-luna"
TAGGING_REASONING_EFFORT = "low"

# Closed taxonomies. Filters compare with `eq`, so these values must be stable.
DOC_TYPES = ("rfp_response", "capabilities", "case_study", "scope_of_work", "other")
SECTORS = (
    "public_health", "transportation", "education",
    "environment", "social_services", "other",
)
TOPICS = (
    "behavior_change", "community_engagement", "paid_media",
    "creative_development", "research_and_evaluation",
    "translation_and_transadaptation", "subcontractor_management",
    "digital_and_social", "crisis_communications", "none",
)
OUTCOMES = ("won", "lost", "unknown")

TAG_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "client": {
            "type": "string",
            "description": "Client or prospective client organization. "
                           "Use 'unknown' if not stated.",
        },
        "agency": {
            "type": "string",
            "description": "Issuing government agency, if different from the "
                           "client. Use 'unknown' if not stated.",
        },
        "year": {
            "type": "number",
            "description": "Four-digit year the document was produced or "
                           "submitted. Use 0 if not determinable.",
        },
        "doc_type": {"type": "string", "enum": list(DOC_TYPES)},
        "sector": {"type": "string", "enum": list(SECTORS)},
        "primary_topic": {"type": "string", "enum": list(TOPICS)},
        "secondary_topic": {"type": "string", "enum": list(TOPICS)},
        "outcome": {
            "type": "string",
            "enum": list(OUTCOMES),
            "description": "Whether the bid was won or lost. Use 'unknown' "
                           "unless the document states it explicitly.",
        },
    },
    "required": [
        "client", "agency", "year", "doc_type",
        "sector", "primary_topic", "secondary_topic", "outcome",
    ],
    "additionalProperties": False,
}

TAGGING_PROMPT = """You are cataloguing business-development documents for a \
marketing agency called Civilian.

Extract metadata for this document. Rules:
- Choose enum values only from the allowed lists.
- `client` is the organization the work is for; `agency` is the issuing body \
if it differs. Use "unknown" rather than guessing.
- `outcome` must be "unknown" unless the document explicitly states the bid \
was won or lost. Do not infer it from tone or content.
- `secondary_topic` may be "none" if the document covers only one topic.
"""


def sha256_of(path: Path) -> str:
    """Content hash, used as the dedupe key.

    Stored as an attribute so the vector store is its own manifest -- there is
    no local state file to drift out of sync with what was actually uploaded.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def coerce_attributes(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Force a tag dict into what the attributes API will accept.

    Drops empty values, coerces types to str/float/bool, truncates over-long
    strings, and caps the key count. Silently exceeding a limit would surface
    later as an opaque 400, so clamp here instead.
    """
    cleaned: Dict[str, Any] = {}

    for key, value in raw.items():
        if value is None or value == "":
            continue
        # year == 0 is the schema's "not determinable" sentinel; omit it so
        # range filters behave sensibly rather than matching a fake year.
        if key == "year" and value in (0, 0.0):
            continue

        key = str(key)[:MAX_ATTRIBUTE_VALUE_CHARS]

        if isinstance(value, bool):
            cleaned[key] = value
        elif isinstance(value, (int, float)):
            cleaned[key] = float(value)
        else:
            cleaned[key] = str(value)[:MAX_ATTRIBUTE_VALUE_CHARS]

        if len(cleaned) >= MAX_ATTRIBUTE_KEYS:
            break

    return cleaned


def build_filter(filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Convert a {key: value} dict from the UI into the API's filter shape.

    Returns None when nothing is selected, which the caller should treat as
    "send no filter at all" rather than "match nothing".
    """
    if not filters:
        return None

    conditions: List[Dict[str, Any]] = []
    for key, value in filters.items():
        if value is None or value == "" or value == "All":
            continue
        if isinstance(value, bool):
            coerced: Any = value
        elif isinstance(value, (int, float)):
            coerced = float(value)
        else:
            coerced = str(value)
        conditions.append({"type": "eq", "key": str(key), "value": coerced})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"type": "and", "filters": conditions}


def tag_document(client, file_id: str, filename: Optional[str] = None) -> Dict[str, Any]:
    """Extract tags for an already-uploaded file.

    The file is read by the model directly via `input_file`, so no local PDF or
    DOCX parsing is needed -- OpenAI extracts text (and page images for PDFs)
    on its side. A file uploaded with purpose="assistants" for the vector store
    can be reused here, so each document is uploaded exactly once.
    """
    import json

    prompt = TAGGING_PROMPT
    if filename:
        prompt += f"\nThe original filename is: {filename}"

    response = client.responses.create(
        model=TAGGING_MODEL,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_file", "file_id": file_id},
                {"type": "input_text", "text": prompt},
            ],
        }],
        text={
            "format": {
                "type": "json_schema",
                "name": "document_tags",
                "schema": TAG_SCHEMA,
                "strict": True,
            }
        },
        reasoning={"effort": TAGGING_REASONING_EFFORT},
    )

    text = getattr(response, "output_text", None)
    if not text:
        raise RuntimeError(f"No tags returned for file {file_id}")

    return json.loads(text)


def attributes_for(tags: Dict[str, Any], content_sha256: str,
                   source_filename: str) -> Dict[str, Any]:
    """Combine model-extracted tags with provenance fields."""
    combined = dict(tags)
    combined["content_sha256"] = content_sha256
    combined["source_filename"] = source_filename
    return coerce_attributes(combined)


def existing_hashes(client, vector_store_id: str) -> Dict[str, str]:
    """Map content_sha256 -> file_id for everything already in the store.

    This is the dedupe manifest. Cheap at corpus sizes in the hundreds; past a
    few thousand files it would want a local cache.
    """
    return existing_fingerprints(client, vector_store_id)[0]


def existing_fingerprints(client, vector_store_id: str):
    """Return (hashes, filenames) for everything already in the store.

    Two dedupe keys because one is not always available. Files uploaded with
    purpose="assistants" cannot be downloaded through the content endpoint, so
    documents tagged by the backfill script have no content hash -- only a
    source_filename. Matching on either prevents re-uploading a document that
    is already indexed.

    Filename matching is the weaker signal (a renamed file slips through), so
    the hash is always preferred when present.
    """
    hashes: Dict[str, str] = {}
    filenames: Dict[str, str] = {}

    for vs_file in client.vector_stores.files.list(vector_store_id=vector_store_id):
        attributes = getattr(vs_file, "attributes", None) or {}
        digest = attributes.get("content_sha256")
        if digest:
            hashes[str(digest)] = vs_file.id
        name = attributes.get("source_filename")
        if name:
            filenames[str(name)] = vs_file.id

    return hashes, filenames
