"""Tests for the password gate.

The production failure was a KeyError at utility.py:15 -- the on_change
callback read st.session_state["password"] with bracket access, in a function
that deletes that same key on success. Any invocation where the key is absent
crashed the whole app with a traceback on the login screen.

Three crash paths existed on that one line: missing session key, missing
secret, and non-ASCII input (hmac.compare_digest rejects non-ASCII str).
All three are covered here.
"""
import pytest
from streamlit.testing.v1 import AppTest

from utility import verify_password


GATED_SCRIPT = """
import streamlit as st
from utility import check_password
if not check_password():
    st.stop()
st.title("Protected content")
"""


def gated(secret="correct-horse"):
    at = AppTest.from_string(GATED_SCRIPT, default_timeout=30)
    if secret is not None:
        at.secrets["password"] = secret
    return at.run()


# ==========================================================================
# verify_password -- must be total: never raises, whatever it is handed
# ==========================================================================

def test_correct_password_accepted():
    assert verify_password("correct-horse", "correct-horse") is True


def test_wrong_password_rejected():
    assert verify_password("wrong", "correct-horse") is False


def test_missing_entered_value_returns_false():
    """The production KeyError: the callback deletes the key it reads, so a
    second invocation sees nothing. Must be a failed login, not a crash."""
    assert verify_password(None, "correct-horse") is False
    assert verify_password("", "correct-horse") is False


def test_missing_expected_value_returns_false():
    """A misconfigured deployment must not crash the login screen."""
    assert verify_password("anything", None) is False
    assert verify_password("anything", "") is False


def test_non_ascii_does_not_raise():
    """hmac.compare_digest rejects non-ASCII str with TypeError. Reproduced
    against the old code; must now be an ordinary failed login."""
    assert verify_password("pässwörd", "correct-horse") is False


def test_non_ascii_password_can_still_match():
    """Comparing as bytes means a non-ASCII password actually works."""
    assert verify_password("pässwörd", "pässwörd") is True


def test_non_string_input_returns_false():
    assert verify_password(12345, "correct-horse") is False


def test_comparison_is_constant_time():
    """Guard against someone 'simplifying' this to ==, which leaks length
    and content through timing."""
    import inspect
    assert "compare_digest" in inspect.getsource(verify_password)


# ==========================================================================
# The gate end to end
# ==========================================================================

def test_gate_blocks_before_authentication():
    at = gated()
    assert [t.label for t in at.text_input] == ["Password"]
    assert not at.title


def test_correct_password_admits():
    at = gated()
    at.text_input[0].set_value("correct-horse").run()
    assert [t.value for t in at.title] == ["Protected content"]


def test_wrong_password_shows_error_not_traceback():
    at = gated()
    at.text_input[0].set_value("wrong").run()
    assert not at.exception
    assert any("incorrect" in e.value.lower() for e in at.error)


def test_non_ascii_input_shows_error_not_traceback():
    """Previously crashed the app with TypeError."""
    at = gated()
    at.text_input[0].set_value("pässwörd").run()
    assert not at.exception, "non-ASCII input crashed the login screen"
    assert any("incorrect" in e.value.lower() for e in at.error)


def test_missing_secret_shows_config_error_not_traceback():
    at = gated(secret=None)
    at.text_input[0].set_value("anything").run()
    assert not at.exception, "missing secret crashed the login screen"
    assert at.error, "no error surfaced to the user"


def test_password_not_retained_after_success():
    """The gate should not keep the plaintext password in session state."""
    at = gated()
    at.text_input[0].set_value("correct-horse").run()
    assert "password" not in at.session_state


def test_already_authenticated_callback_is_a_noop():
    """A stray second callback after login must not revoke access. This is
    the shape of the production failure: the callback fires again once the
    key it reads has been removed."""
    at = gated()
    at.text_input[0].set_value("correct-horse").run()
    assert at.session_state["password_correct"] is True

    at.run()
    assert not at.exception
    assert at.session_state["password_correct"] is True
    assert [t.value for t in at.title] == ["Protected content"]
