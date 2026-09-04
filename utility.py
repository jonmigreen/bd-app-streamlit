# filename: utility.py
"""Shared Streamlit components. Currently the password gate."""
import hmac

import streamlit as st

# Session state keys owned by the gate
_PASSWORD_KEY = "password"          # the text_input widget's key
_AUTHENTICATED_KEY = "password_correct"
_CONFIG_ERROR_KEY = "password_config_error"


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

    def password_entered():
        """on_change handler for the password field."""
        # Ignore stray callbacks once authenticated. Without this, a second
        # invocation after a successful login -- the shape of the production
        # failure -- would revoke access by flipping the flag back to False.
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
            # Don't retain the plaintext password. pop() rather than del, so a
            # repeat call cannot raise.
            st.session_state.pop(_PASSWORD_KEY, None)
        else:
            st.session_state[_AUTHENTICATED_KEY] = False

    if st.session_state.get(_AUTHENTICATED_KEY, False):
        return True

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
