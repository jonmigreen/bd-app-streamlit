#!/usr/bin/env python
"""Upload a folder of documents to the vector store, tagging each one.

    # Always start here -- tags are computed and printed, nothing is stored.
    ./venv/bin/python scripts/ingest_documents.py --folder ./documents --dry-run

    # Then one real file, to confirm attributes actually attach.
    ./venv/bin/python scripts/ingest_documents.py --folder ./documents --limit 1

    ./venv/bin/python scripts/ingest_documents.py --folder ./documents

Re-running is safe: each file's content hash is stored as an attribute, so
already-ingested documents are skipped.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI  # noqa: E402

from config import Config  # noqa: E402
from document_tagger import (  # noqa: E402
    attributes_for,
    existing_fingerprints,
    sha256_of,
    tag_document,
)

# Document formats the vector store accepts. The retrieval API also supports
# source-code extensions; those are excluded as noise for this corpus.
SUPPORTED_SUFFIXES = {
    ".pdf", ".doc", ".docx", ".pptx", ".txt", ".md", ".html", ".json",
}

MAX_FILE_BYTES = 512 * 1024 * 1024  # API limit


def discover(folder: Path):
    """Supported files under folder, sorted for deterministic runs."""
    return sorted(
        path for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
        and not path.name.startswith((".", "~$"))
    )


def fallback_tags(path: Path) -> dict:
    """Minimal tags when --skip-tagging is used."""
    return {
        "client": "unknown",
        "agency": "unknown",
        "doc_type": "other",
        "sector": "other",
        "primary_topic": "none",
        "secondary_topic": "none",
        "outcome": "unknown",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true",
                        help="Tag and print without adding to the vector store. "
                             "Files are uploaded for tagging, then deleted.")
    parser.add_argument("--skip-tagging", action="store_true",
                        help="Upload without calling the tagging model.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not args.folder.is_dir():
        print(f"Not a directory: {args.folder}")
        return 1

    Config.validate()
    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    store_id = Config.OPENAI_VECTOR_STORE_ID

    files = discover(args.folder)
    if args.limit:
        files = files[:args.limit]

    if not files:
        print(f"No supported documents found under {args.folder}")
        print(f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}")
        return 0

    print("=" * 76)
    print(f"{'DRY RUN -- nothing will be stored' if args.dry_run else 'INGEST'}")
    print("=" * 76)
    print(f"  folder       : {args.folder}")
    print(f"  vector store : {store_id}")
    print(f"  documents    : {len(files)}")
    print("-" * 76)

    print("  reading existing store contents for dedupe...")
    try:
        seen, seen_names = existing_fingerprints(client, store_id)
    except Exception as exc:
        print(f"  could not list store ({type(exc).__name__}: {exc})")
        return 1
    print(f"  {len(seen)} hashed, {len(seen_names)} named file(s) already in store")
    print("-" * 76)

    ingested = skipped = failed = 0

    for path in files:
        rel = path.relative_to(args.folder)
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            print(f"  SKIP  {rel}  (exceeds 512 MB)")
            skipped += 1
            continue

        digest = sha256_of(path)
        if digest in seen:
            print(f"  SKIP  {rel}  (already ingested)")
            skipped += 1
            continue
        # Backfilled documents have no hash (assistants-purpose files cannot
        # be downloaded), so fall back to matching the original filename.
        if path.name in seen_names:
            print(f"  SKIP  {rel}  (filename already in store)")
            skipped += 1
            continue

        uploaded = None
        try:
            with open(path, "rb") as handle:
                uploaded = client.files.create(file=handle, purpose="assistants")

            if args.skip_tagging:
                tags = fallback_tags(path)
            else:
                tags = tag_document(client, uploaded.id, filename=path.name)

            attributes = attributes_for(tags, digest, path.name)

            if args.dry_run:
                # Leave no trace: the upload existed only so the model could
                # read the document.
                client.files.delete(uploaded.id)
                print(f"  TAGS  {rel}")
                for key, value in sorted(attributes.items()):
                    if key != "content_sha256":
                        print(f"          {key:18s} {value}")
            else:
                client.vector_stores.files.create_and_poll(
                    vector_store_id=store_id,
                    file_id=uploaded.id,
                    attributes=attributes,
                )
                seen[digest] = uploaded.id
                seen_names[path.name] = uploaded.id
                summary = " | ".join(
                    f"{attributes.get(k)}" for k in ("client", "year", "doc_type")
                    if attributes.get(k) is not None
                )
                print(f"  OK    {rel}  [{summary}]")
            ingested += 1

        except Exception as exc:
            failed += 1
            print(f"  FAIL  {rel}  ({type(exc).__name__}: {str(exc)[:120]})")
            if uploaded is not None:
                try:
                    client.files.delete(uploaded.id)
                except Exception:
                    pass

    print("-" * 76)
    label = "would ingest" if args.dry_run else "ingested"
    print(f"  {label}: {ingested}   skipped: {skipped}   failed: {failed}")
    print("=" * 76)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
