#!/usr/bin/env python
"""Tag documents already present in the vector store.

    ./venv/bin/python scripts/backfill_tags.py --dry-run
    ./venv/bin/python scripts/backfill_tags.py --only-untagged

Nothing is re-uploaded and nothing is re-embedded: a vector store file's id is
a Files API file id, so the model reads it in place via `input_file`, and
`vector_stores.files.update()` attaches the resulting attributes. The only cost
is the tagging call itself.

File bytes are downloaded solely to compute the content hash, so that documents
backfilled here are recognised as duplicates if the same file is later fed to
ingest_documents.py.
"""
import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI  # noqa: E402

from config import Config  # noqa: E402
from document_tagger import attributes_for, tag_document  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute and print tags without writing them.")
    parser.add_argument("--only-untagged", action="store_true",
                        help="Skip files that already have attributes.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    Config.validate()
    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    store_id = Config.OPENAI_VECTOR_STORE_ID

    print("=" * 76)
    print("DRY RUN -- no attributes will be written" if args.dry_run else "BACKFILL TAGS")
    print("=" * 76)
    print(f"  vector store : {store_id}")
    print("-" * 76)

    try:
        vs_files = list(client.vector_stores.files.list(vector_store_id=store_id))
    except Exception as exc:
        print(f"  could not list store ({type(exc).__name__}: {exc})")
        return 1

    if args.limit:
        vs_files = vs_files[:args.limit]
    print(f"  {len(vs_files)} file(s) in store")
    print("-" * 76)

    tagged = skipped = failed = 0

    for vs_file in vs_files:
        file_id = vs_file.id
        existing = getattr(vs_file, "attributes", None) or {}

        if args.only_untagged and existing:
            print(f"  SKIP  {file_id}  (already has {len(existing)} attribute(s))")
            skipped += 1
            continue

        try:
            try:
                filename = client.files.retrieve(file_id).filename
            except Exception:
                filename = file_id

            # Hash the original bytes so this file dedupes against a future
            # local ingest of the same document. Files uploaded with
            # purpose="assistants" usually cannot be downloaded, in which case
            # source_filename becomes the dedupe key instead.
            digest = ""
            try:
                raw = client.files.content(file_id).read()
                digest = hashlib.sha256(raw).hexdigest()
            except Exception:
                pass

            tags = tag_document(client, file_id, filename=filename)
            attributes = attributes_for(tags, digest, filename)
            if not digest:
                attributes.pop("content_sha256", None)

            if args.dry_run:
                print(f"  TAGS  {filename}")
                for key, value in sorted(attributes.items()):
                    if key != "content_sha256":
                        print(f"          {key:18s} {value}")
            else:
                client.vector_stores.files.update(
                    vector_store_id=store_id,
                    file_id=file_id,
                    attributes=attributes,
                )
                summary = " | ".join(
                    str(attributes.get(k)) for k in ("client", "year", "doc_type")
                    if attributes.get(k) is not None
                )
                print(f"  OK    {filename}  [{summary}]")
            tagged += 1

        except Exception as exc:
            failed += 1
            print(f"  FAIL  {file_id}  ({type(exc).__name__}: {str(exc)[:120]})")

    print("-" * 76)
    label = "would tag" if args.dry_run else "tagged"
    print(f"  {label}: {tagged}   skipped: {skipped}   failed: {failed}")
    print("=" * 76)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
