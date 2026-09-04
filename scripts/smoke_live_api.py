#!/usr/bin/env python
"""Tier 3: one real OpenAI call to validate the API contract.

The pytest suite mocks the SDK, so by construction it cannot catch
"OpenAI rejects this parameter". This script can. It is the only check that
proves gpt-5.6-terra actually accepts reasoning={"effort": ...} alongside
file_search with ranking_options.

Costs a few cents. Run manually; never in CI.

    ./venv/bin/python scripts/smoke_live_api.py
    ./venv/bin/python scripts/smoke_live_api.py --query "your question" --threshold 0.4
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai_client import MAX_SEARCH_RESULTS, OpenAIClient  # noqa: E402
from config import Config  # noqa: E402

DEFAULT_QUERY = "Summarize Civilian's approach to campaign development."


def check_research_path():
    """Verify the model still emits the [Source N] format Research mode parses.

    Chat mode uses OpenAI's native citation annotations, but ask_about_chunks
    relies on the model following a prompt instruction to write [Source N].
    If the model drifts to another format, cited_sources silently falls back
    to "all chunks" -- wrong, but indistinguishable from working.
    """
    client = OpenAIClient()
    chunks = [
        {"filename": f"doc_{i}.pdf",
         "content": f"Document {i} discusses topic number {i} in detail."}
        for i in range(1, 13)
    ]
    question = "Which document discusses topic number 12? Cite your source."

    print("=" * 68)
    print("RESEARCH PATH CHECK  ([Source N] citation format)")
    print("=" * 68)
    print(f"  model   : {client.model}")
    print(f"  chunks  : {len(chunks)}")
    print("-" * 68)

    try:
        answer, cited = client.ask_about_chunks(question, chunks)
    except Exception as exc:
        print(f"\n  FAILED: {type(exc).__name__}: {exc}\n")
        return 1

    import re
    found = re.findall(r"\[Source (\d+)\]", answer or "")
    print(f"  [Source N] markers found : {found or 'NONE'}")
    print(f"  chunks reported as cited : {[c['filename'] for c in cited]}")
    if not found:
        print("\n  WARNING: no [Source N] markers. cited_sources fell back to")
        print("  ALL chunks, which looks correct but attributes nothing.")
    elif len(cited) == len(chunks):
        print("\n  WARNING: every chunk reported as cited -- check the fallback.")
    else:
        print("\n  OK: citation parsing resolved a specific subset.")
    print("-" * 68)
    print("  ANSWER:")
    print("  " + (answer or "").strip()[:400].replace("\n", "\n  "))
    print("=" * 68)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument(
        "--research",
        action="store_true",
        help="Instead of the Chat path, check that the Research path still "
             "emits [Source N] citations (what ask_about_chunks parses).",
    )
    args = parser.parse_args()

    if args.research:
        return check_research_path()

    print("=" * 68)
    print("LIVE API SMOKE TEST  (makes one real, billable call)")
    print("=" * 68)
    print(f"  model      : {Config.OPENAI_MODEL}")
    print(f"  effort     : {Config.OPENAI_REASONING_EFFORT}")
    print(f"  vector store: {Config.OPENAI_VECTOR_STORE_ID}")
    print(f"  threshold  : {args.threshold}")
    print(f"  query      : {args.query}")
    print("-" * 68)

    client = OpenAIClient()

    try:
        response = client.client.responses.create(
            model=client.model,
            input=args.query,
            tools=[{
                "type": "file_search",
                "vector_store_ids": [client.vector_store_id],
                "max_num_results": MAX_SEARCH_RESULTS,
                "ranking_options": {
                    "ranker": "auto",
                    "score_threshold": args.threshold,
                },
            }],
            include=["file_search_call.results"],
            reasoning={"effort": client.reasoning_effort},
        )
    except Exception as exc:
        print(f"\n  FAILED: {type(exc).__name__}: {exc}\n")
        print("  If this is a 400, the model likely rejects one of:")
        print("    reasoning={'effort': ...} / ranking_options / the model id")
        return 1

    print(f"  model echoed  : {getattr(response, 'model', '(none)')}")

    text, citations = client._extract_citations_from_response(response)
    print(f"  output text   : {len(text or '')} chars")
    print(f"  citations     : {len(citations)}")

    results = client._extract_file_search_results(response)
    print(f"  chunks returned: {len(results)}")
    scores = sorted(
        (r["score"] for r in results.values() if r.get("score") is not None),
        reverse=True,
    )
    if scores:
        print(f"  score range   : {scores[0]:.3f} (max) .. {scores[-1]:.3f} (min)")
        below = [s for s in scores if s < args.threshold]
        print(f"  below threshold: {len(below)}  (expect 0 if the ranker honours it)")
    else:
        print("  score range   : no scores returned")

    usage = getattr(response, "usage", None)
    if usage:
        out_details = getattr(usage, "output_tokens_details", None)
        reasoning_tokens = getattr(out_details, "reasoning_tokens", None)
        print(f"  input tokens  : {getattr(usage, 'input_tokens', '?')}")
        print(f"  output tokens : {getattr(usage, 'output_tokens', '?')}"
              f"  (reasoning: {reasoning_tokens if reasoning_tokens is not None else '?'})")

    print("-" * 68)
    print("  ANSWER PREVIEW (check it still uses [Source N] citation format):")
    preview = (text or "")[:600]
    print("  " + preview.replace("\n", "\n  "))
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
