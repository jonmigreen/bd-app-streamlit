"""Tests for Config loading and validation."""
import pytest

from config import Config


def test_defaults_are_current_generation():
    assert Config.OPENAI_MODEL == "gpt-5.6-terra"
    assert Config.OPENAI_REASONING_EFFORT == "low"


def test_validate_passes_with_full_config():
    assert Config.validate() is True


def test_validate_requires_api_key(monkeypatch):
    monkeypatch.setattr(Config, "OPENAI_API_KEY", None)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        Config.validate()


def test_validate_requires_vector_store_id(monkeypatch):
    monkeypatch.setattr(Config, "OPENAI_VECTOR_STORE_ID", None)
    with pytest.raises(ValueError, match="OPENAI_VECTOR_STORE_ID"):
        Config.validate()


@pytest.mark.parametrize(
    "effort", ["none", "low", "medium", "high", "xhigh", "max"]
)
def test_all_documented_efforts_accepted(monkeypatch, effort):
    monkeypatch.setattr(Config, "OPENAI_REASONING_EFFORT", effort)
    assert Config.validate() is True


@pytest.mark.parametrize("effort", ["bogus", "LOW", "0.7", "", "minimal"])
def test_invalid_effort_rejected(monkeypatch, effort):
    """Must fail fast with ValueError rather than 400ing at the API.

    app.py:48 catches ValueError specifically, so this surfaces as a clean
    Streamlit error instead of a traceback.
    """
    monkeypatch.setattr(Config, "OPENAI_REASONING_EFFORT", effort)
    with pytest.raises(ValueError, match="OPENAI_REASONING_EFFORT"):
        Config.validate()
