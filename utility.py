# filename: utility.py
"""Shared Streamlit components. Currently the password gate."""
import hmac
import time

import streamlit as st

from logger import api_error_logger

# Session state keys owned by the gate
_PASSWORD_KEY = "password"          # the text_input widget's key
_AUTHENTICATED_KEY = "password_correct"
_CONFIG_ERROR_KEY = "password_config_error"

# Rate limit tuning. Per-client is the first line; global is the backstop an
# attacker cannot sidestep by rotating identity (see _client_key).
_CLIENT_MAX_ATTEMPTS = 5
_CLIENT_WINDOW_SECONDS = 900        # 15 minutes
_CLIENT_BASE_LOCKOUT = 30
_CLIENT_MAX_LOCKOUT = 900

_GLOBAL_MAX_ATTEMPTS = 25
_GLOBAL_WINDOW_SECONDS = 900
_GLOBAL_BASE_LOCKOUT = 60
_GLOBAL_MAX_LOCKOUT = 1800


class RateLimiter:
    """Sliding-window failure counter with exponential lockout.

    State lives in the process, not in st.session_state: session state is
    per-connection, so an attacker resets it just by reconnecting. Process
    state survives that, though it is still lost when the app restarts or is
    put to sleep, and would not be shared across replicas.
    """

    def __init__(self, max_attempts, window_seconds, base_lockout, max_lockout):
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self.base_lockout = base_lockout
        self.max_lockout = max_lockout
        self._failures = {}   # key -> [timestamps]

    def _recent(self, key, now):
        stamps = [t for t in self._failures.get(key, []) if now - t < self.window_seconds]
        if stamps:
            self._failures[key] = stamps
        else:
            self._failures.pop(key, None)
        return stamps

    def record_failure(self, key, now=None):
        now = time.time() if now is None else now
        stamps = self._recent(key, now)
        stamps.append(now)
        self._failures[key] = stamps
        return len(stamps)

    def remaining(self, key, now=None):
        """Seconds until this key may try again; 0 when not locked."""
        now = time.time() if now is None else now
        stamps = self._recent(key, now)
        if len(stamps) < self.max_attempts:
            return 0

        # Each failure past the threshold doubles the wait, so sustained
        # guessing becomes impractical rather than merely slow.
        over = len(stamps) - self.max_attempts
        lockout = min(self.base_lockout * (2 ** over), self.max_lockout)
        elapsed = now - stamps[-1]
        return max(0, lockout - elapsed)

    def reset(self, key):
        self._failures.pop(key, None)


_client_limiter = RateLimiter(
    _CLIENT_MAX_ATTEMPTS, _CLIENT_WINDOW_SECONDS,
    _CLIENT_BASE_LOCKOUT, _CLIENT_MAX_LOCKOUT,
)
_global_limiter = RateLimiter(
    _GLOBAL_MAX_ATTEMPTS, _GLOBAL_WINDOW_SECONDS,
    _GLOBAL_BASE_LOCKOUT, _GLOBAL_MAX_LOCKOUT,
)


def reset_rate_limits():
    """Clear all counters. For tests, and for an admin recovering from a lockout."""
    _client_limiter._failures.clear()
    _global_limiter._failures.clear()


def _record_failed_attempt(client, now=None):
    """Count a failure against the global limiter, and the per-client one
    only when the client is actually distinguishable."""
    _global_limiter.record_failure("all", now=now)
    if client != "unknown":
        _client_limiter.record_failure(client, now=now)


def _lockout_seconds(client, now=None):
    """Seconds this visitor must wait, 0 if not locked.

    The per-client limiter is skipped for an unresolvable client. Behind a
    proxy every visitor shares one key, so enforcing the low per-client
    threshold there would let any handful of failures -- from any mix of
    people -- lock out everyone, while the global backstop never binds. A
    limiter keyed on a value identical for all users is not per-client at
    all; it is a mislabelled global limiter with the wrong threshold.
    """
    wait = _global_limiter.remaining("all", now=now)
    if client != "unknown":
        wait = max(wait, _client_limiter.remaining(client, now=now))
    return wait


def _client_key():
    """Best-effort client identifier.

    Streamlit documents ip_address as spoofable and explicitly not a security
    control, so this only separates honest users from each other. The global
    limiter is what actually bounds an attacker.
    """
    try:
        ip = st.context.ip_address
    except Exception:
        return "unknown"
    # Must be a real, stable string. Anything else -- None, or a test double
    # that returns a fresh object each call -- would hand every attempt its
    # own bucket and silently disable per-client limiting.
    if isinstance(ip, str) and ip.strip():
        return ip.strip()
    return "unknown"


def _format_wait(seconds):
    seconds = int(seconds) + 1
    if seconds < 60:
        return f"{seconds} second{'s' if seconds != 1 else ''}"
    minutes = (seconds + 59) // 60
    return f"{minutes} minute{'s' if minutes != 1 else ''}"


def verify_password(entered, expected) -> bool:
    """Constant-time password comparison that never raises.

    This runs inside a Streamlit on_change callback, where an unhandled
    exception takes down the whole page with a traceback instead of showing a
    login error. It therefore has to tolerate every input it might be handed:

    - `entered` missing -- the callback can fire when the widget's session
      state key is absent, which is what crashed production with a KeyError.
    - `expected` missing -- a deployment whose `password` secret is not set.
    - Non-ASCII text -- hmac.compare_digest raises TypeError on non-ASCII
      `str`, so both sides are encoded to bytes first. That also means a
      non-ASCII password genuinely works rather than erroring.

    Comparison stays constant-time: `==` would leak the password through
    response timing.
    """
    if not entered or not expected:
        return False
    if not isinstance(entered, str) or not isinstance(expected, str):
        return False
    try:
        return hmac.compare_digest(entered.encode("utf-8"), expected.encode("utf-8"))
    except Exception:
        return False


def check_password() -> bool:
    """Returns True once the user has entered the correct password."""

    client = _client_key()

    def password_entered():
        """on_change handler for the password field."""
        # Ignore stray callbacks once authenticated. Without this, a second
        # invocation after a successful login -- the shape of the production
        # KeyError -- would revoke access by flipping the flag back to False.
        if st.session_state.get(_AUTHENTICATED_KEY, False):
            return

        # .get(), never [...]: this function removes _PASSWORD_KEY on success,
        # so it cannot assume the key it reads still exists.
        entered = st.session_state.get(_PASSWORD_KEY, "")

        try:
            expected = st.secrets[_PASSWORD_KEY]
        except Exception:
            # Misconfigured deployment. Surface it as a message, not a crash.
            st.session_state[_CONFIG_ERROR_KEY] = True
            st.session_state[_AUTHENTICATED_KEY] = False
            return

        if verify_password(entered, expected):
            st.session_state[_AUTHENTICATED_KEY] = True
            st.session_state[_CONFIG_ERROR_KEY] = False
            _client_limiter.reset(client)
            # Don't retain the plaintext password. pop() rather than del, so a
            # repeat call cannot raise.
            st.session_state.pop(_PASSWORD_KEY, None)
        else:
            st.session_state[_AUTHENTICATED_KEY] = False
            _record_failed_attempt(client)
            # Failed logins previously left no trace anywhere.
            api_error_logger.error(f"Failed login attempt from client={client}")
            st.session_state.pop(_PASSWORD_KEY, None)

    if st.session_state.get(_AUTHENTICATED_KEY, False):
        return True

    wait = _lockout_seconds(client)
    if wait > 0:
        # Render no input at all: there is nothing to submit while locked, so
        # guessing cannot continue. Refuse rather than sleep -- sleeping would
        # hold a server thread and turn this into a denial-of-service lever.
        st.error(
            f"🔒 Too many failed attempts. Please try again in {_format_wait(wait)}."
        )
        return False

    st.text_input(
        "Password", type="password", on_change=password_entered, key=_PASSWORD_KEY
    )

    if st.session_state.get(_CONFIG_ERROR_KEY):
        st.error(
            "🔒 This app's password is not configured. "
            "Please contact the app administrator."
        )
    elif st.session_state.get(_AUTHENTICATED_KEY) is False:
        st.error("😕 Password incorrect")

    return False
