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


# ==========================================================================
# Rate limiting
#
# Streamlit's own docs say st.context.ip_address "should not be used for
# security measures because it can easily be spoofed", so the per-client
# bucket is only a first line. The global bucket is the part an attacker
# cannot sidestep by rotating identity.
# ==========================================================================

from utility import RateLimiter, reset_rate_limits  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_limiters():
    reset_rate_limits()
    yield
    reset_rate_limits()


def limiter(**kw):
    defaults = dict(max_attempts=3, window_seconds=60,
                    base_lockout=10, max_lockout=100)
    defaults.update(kw)
    return RateLimiter(**defaults)


def test_no_lockout_before_threshold():
    rl = limiter()
    rl.record_failure("a", now=0)
    rl.record_failure("a", now=1)
    assert rl.remaining("a", now=2) == 0


def test_lockout_triggers_at_threshold():
    rl = limiter()
    for i in range(3):
        rl.record_failure("a", now=i)
    assert rl.remaining("a", now=3) > 0


def test_lockout_expires():
    rl = limiter()
    for i in range(3):
        rl.record_failure("a", now=i)
    assert rl.remaining("a", now=1000) == 0


def test_lockout_grows_exponentially():
    """Each failure past the threshold should cost more, so sustained
    guessing becomes impractical rather than merely slow."""
    rl = limiter()
    for i in range(3):
        rl.record_failure("a", now=i)
    first = rl.remaining("a", now=3)

    rl.record_failure("a", now=4)
    second = rl.remaining("a", now=5)

    assert second > first


def test_lockout_is_capped():
    rl = limiter(max_lockout=100)
    for i in range(30):
        rl.record_failure("a", now=i)
    assert rl.remaining("a", now=31) <= 100


def test_old_failures_fall_out_of_window():
    """Two failures today and one next week is not an attack."""
    rl = limiter()
    rl.record_failure("a", now=0)
    rl.record_failure("a", now=1)
    rl.record_failure("a", now=10_000)
    assert rl.remaining("a", now=10_001) == 0


def test_buckets_are_independent():
    rl = limiter()
    for i in range(3):
        rl.record_failure("a", now=i)
    assert rl.remaining("b", now=4) == 0


def test_success_resets_the_bucket():
    rl = limiter()
    rl.record_failure("a", now=0)
    rl.record_failure("a", now=1)
    rl.reset("a")
    rl.record_failure("a", now=2)
    assert rl.remaining("a", now=3) == 0


# --- the gate, end to end -------------------------------------------------

@pytest.fixture
def resolvable_client(monkeypatch):
    """Give the visitor a real, distinct IP so per-client limiting applies."""
    import utility
    from conftest import ns
    monkeypatch.setattr(utility.st, "context", ns(ip_address="203.0.113.7"))


def _guess(at, times, password="wrong"):
    for _ in range(times):
        if not at.text_input:
            break          # locked out; the field is gone
        at.text_input[0].set_value(password).run()


def test_repeated_wrong_passwords_lock_the_form(resolvable_client):
    at = gated()
    _guess(at, 6)

    assert not at.exception
    blob = " ".join(e.value for e in at.error).lower()
    assert "too many" in blob or "try again" in blob


def test_lockout_hides_the_password_input(resolvable_client):
    """While locked there is nothing to submit, so guessing cannot continue."""
    at = gated()
    _guess(at, 6)

    assert not at.text_input, "password field still accepting input while locked"


def test_shared_client_key_does_not_lock_out_early():
    """The regression this guards: behind a proxy every visitor shares a key,
    so a handful of failures must NOT lock out the whole team."""
    at = gated()
    _guess(at, 8)

    assert at.text_input, "shared-key visitors locked out well below the global threshold"


def test_correct_password_still_works_below_threshold():
    at = gated()
    at.text_input[0].set_value("wrong").run()
    at.text_input[0].set_value("correct-horse").run()

    assert [t.value for t in at.title] == ["Protected content"]


def test_client_key_is_stable_when_ip_is_not_a_string(monkeypatch):
    """CI caught this: st.context.ip_address returned a MagicMock, so every
    attempt got a unique bucket key and per-client limiting did nothing.
    Any non-string must collapse to one shared key."""
    from unittest.mock import MagicMock

    import utility

    monkeypatch.setattr(utility.st, "context", MagicMock())

    assert utility._client_key() == "unknown"
    assert utility._client_key() == utility._client_key()


def test_client_key_uses_a_real_ip(monkeypatch):
    import utility
    from conftest import ns

    monkeypatch.setattr(utility.st, "context", ns(ip_address=" 203.0.113.7 "))
    assert utility._client_key() == "203.0.113.7"


def test_client_key_falls_back_when_ip_is_none(monkeypatch):
    import utility
    from conftest import ns

    monkeypatch.setattr(utility.st, "context", ns(ip_address=None))
    assert utility._client_key() == "unknown"


# --- per-client limiting must not become a too-strict global limiter ------

def test_unresolvable_client_is_not_rate_limited_per_client(monkeypatch):
    """Behind Streamlit Cloud's proxy every visitor shares one key. Enforcing
    the per-client threshold on that shared key would let any 5 failures --
    from any mix of people -- lock out the whole team, while the real global
    backstop (25) never binds."""
    from unittest.mock import MagicMock

    import utility

    monkeypatch.setattr(utility.st, "context", MagicMock())
    reset_rate_limits()

    key = utility._client_key()
    assert key == "unknown"

    for i in range(10):
        utility._record_failed_attempt(key, now=1000 + i)

    assert utility._lockout_seconds(key, now=1010) == 0, (
        "shared key should be governed by the global limiter, not per-client"
    )


def test_resolvable_clients_are_limited_independently(monkeypatch):
    import utility

    reset_rate_limits()
    for i in range(6):
        utility._record_failed_attempt("203.0.113.7", now=1000 + i)

    assert utility._lockout_seconds("203.0.113.7", now=1006) > 0
    assert utility._lockout_seconds("198.51.100.4", now=1006) == 0


def test_global_backstop_still_applies_to_shared_key(monkeypatch):
    """Unresolvable clients are still bounded -- just by the global limiter."""
    import utility

    reset_rate_limits()
    for i in range(26):
        utility._record_failed_attempt("unknown", now=1000 + i)

    assert utility._lockout_seconds("unknown", now=1026) > 0
