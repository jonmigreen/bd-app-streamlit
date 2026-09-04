"""Front-end tests driven headlessly by streamlit.testing.v1.AppTest.

No browser and no network: the OpenAI client is a FakeClient pre-seeded into
session_state, which app.py's `if "openai_client" not in st.session_state`
guard leaves untouched.

Pages are rendered by importing app and calling the page function directly.
AppTest cannot navigate st.navigation/st.Page apps, and app.py's main() guard
makes the module importable without rendering the whole app.
"""
import pytest
from streamlit.testing.v1 import AppTest

from conftest import FakeClient, make_snippet


def run_page(page_fn, fake, **session):
    """Render one page function under AppTest with a fake client."""
    script = f"import app\napp.init_session_state()\napp.{page_fn}()"
    at = AppTest.from_string(script, default_timeout=30)
    at.secrets["password"] = "test-password"
    at.session_state["password_correct"] = True
    at.session_state["openai_client"] = fake
    for key, value in session.items():
        at.session_state[key] = value
    return at.run()


def click(at, label_fragment):
    """Click the first button whose label contains the fragment."""
    for button in at.button:
        if label_fragment in button.label:
            return button.click()
    raise AssertionError(
        f"No button matching {label_fragment!r}; saw {[b.label for b in at.button]}"
    )


# ==========================================================================
# Bug 4.3: a new search must not inherit the previous search's selections
# ==========================================================================

def test_new_search_clears_snippet_selections():
    """Fails on main: keyed widgets persisted, silently attaching stale
    snippets from search A to a question asked about search B."""
    fake = FakeClient(results=[make_snippet(i) for i in range(3)])
    at = run_page("research_page", fake)

    # First search
    at.text_input[0].set_value("term A")
    click(at, "Search").run()
    assert at.session_state["search_generation"] == 1

    # Select two snippets
    at.checkbox(key="snippet_1_0").check()
    at.checkbox(key="snippet_1_1").check()
    at.run()
    assert sorted(at.session_state["selected_snippets"]) == [0, 1]

    # Second, unrelated search
    at.text_input[0].set_value("term B")
    click(at, "Search").run()

    assert at.session_state["search_generation"] == 2
    assert at.session_state["selected_snippets"] == []
    assert not any(
        at.checkbox(key=f"snippet_2_{i}").value for i in range(3)
    ), "checkboxes carried over into the new result set"


def test_search_generation_increments_once_per_search():
    fake = FakeClient(results=[make_snippet(i) for i in range(2)])
    at = run_page("research_page", fake)

    assert at.session_state["search_generation"] == 0
    for expected in (1, 2, 3):
        at.text_input[0].set_value(f"query {expected}")
        click(at, "Search").run()
        assert at.session_state["search_generation"] == expected


def test_selecting_a_snippet_records_its_index():
    fake = FakeClient(results=[make_snippet(i) for i in range(3)])
    at = run_page("research_page", fake)
    at.text_input[0].set_value("q")
    click(at, "Search").run()

    at.checkbox(key="snippet_1_2").check()
    at.run()

    assert at.session_state["selected_snippets"] == [2]


def test_empty_query_warns_and_does_not_search():
    fake = FakeClient(results=[make_snippet(0)])
    at = run_page("research_page", fake)

    click(at, "Search").run()

    assert any("enter a search query" in w.value.lower() for w in at.warning)
    assert fake.search_calls == []


# ==========================================================================
# Bug 4.1: the sidebar slider must reach the client layer
# ==========================================================================

def test_slider_value_reaches_client_on_chat():
    """The UI linkage Bug 4.1 broke: slider -> get_rag_response threshold."""
    fake = FakeClient(answer="Grounded answer.")
    at = run_page("chat_page", fake)

    at.slider[0].set_value(0.85).run()
    at.chat_input[0].set_value("What is our approach?").run()

    assert len(fake.rag_calls) == 1
    assert fake.rag_calls[0]["min_relevance_score"] == 0.85


def test_slider_value_reaches_client_on_research():
    fake = FakeClient(results=[make_snippet(0)])
    at = run_page("research_page", fake)

    at.slider[0].set_value(0.35).run()
    at.text_input[0].set_value("fiscal management")
    click(at, "Search").run()

    assert fake.search_calls[0]["min_relevance_score"] == 0.35


def test_debug_readout_reports_threshold_not_hardcoded_zero():
    """The reworded debug line must reflect the real threshold."""
    fake = FakeClient(answer="A.", sources=[make_snippet(1), make_snippet(2)])
    at = run_page("chat_page", fake, debug_mode=True)

    at.slider[0].set_value(0.6).run()
    at.chat_input[0].set_value("question").run()

    info_text = " ".join(i.value for i in at.info)
    assert "0.60" in info_text
    assert "2 source(s)" in info_text


# ==========================================================================
# Chat behaviour
# ==========================================================================

def test_answer_and_question_appended_to_history():
    fake = FakeClient(answer="The answer.")
    at = run_page("chat_page", fake)

    at.chat_input[0].set_value("A question?").run()

    messages = at.session_state["messages"]
    assert messages[0] == {"role": "user", "content": "A question?"}
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "The answer."


def test_api_error_renders_error_not_traceback():
    """A client failure must surface as a Streamlit error, not a crash."""
    fake = FakeClient(error=RuntimeError("upstream exploded"))
    at = run_page("chat_page", fake)

    at.chat_input[0].set_value("question").run()

    assert not at.exception, "error escaped as an unhandled exception"
    assert len(at.error) >= 1


def test_config_error_is_reported_distinctly():
    """ValueError is handled separately from generic failures (app.py:289)."""
    fake = FakeClient(error=ValueError("OPENAI_API_KEY is not set"))
    at = run_page("chat_page", fake)

    at.chat_input[0].set_value("question").run()

    assert not at.exception
    assert any("Configuration Error" in e.value for e in at.error)


# ==========================================================================
# Password gate (utility.check_password)
# ==========================================================================

GATED_SCRIPT = """
import streamlit as st
from utility import check_password
if not check_password():
    st.stop()
st.title("Protected content")
"""


def test_password_gate_blocks_unauthenticated():
    at = AppTest.from_string(GATED_SCRIPT, default_timeout=30)
    at.secrets["password"] = "correct-horse"
    at.run()

    assert [t.label for t in at.text_input] == ["Password"]
    assert not at.title, "page content rendered before authentication"


def test_password_gate_admits_correct_password():
    at = AppTest.from_string(GATED_SCRIPT, default_timeout=30)
    at.secrets["password"] = "correct-horse"
    at.run()

    at.text_input[0].set_value("correct-horse").run()

    assert [t.value for t in at.title] == ["Protected content"]


def test_password_gate_rejects_wrong_password():
    at = AppTest.from_string(GATED_SCRIPT, default_timeout=30)
    at.secrets["password"] = "correct-horse"
    at.run()

    at.text_input[0].set_value("wrong").run()

    assert not at.title
    assert any("incorrect" in e.value.lower() for e in at.error)
