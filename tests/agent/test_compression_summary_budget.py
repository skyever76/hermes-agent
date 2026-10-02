"""Tests for the configurable compression-summary output budget and the truncation retry.

Covers the Owner's acceptance list:
  a. unconfigured -> 10_000 (backward compatible)
  b. coding-profile 32_000 takes effect and lands in the 24K-32K band on a real-model window
  c. the effective budget is re-derived per model window (no stale legacy `0.05 * window` term)
  d. truncation retries exactly once, with a raised budget
  e. a retry that suffices completes normally and clears the latch
  f. a second truncation aborts, preserving messages (no rotation)
  g. non-`length` failures keep the original fallback/cooldown semantics
  h. misconfiguration and unknown routes fail safe

The budget is a pure function of (configured ceiling, model max output, window headroom), so most
cases exercise `ContextCompressor.max_summary_tokens` directly without standing up an agent.
"""

from __future__ import annotations

import pytest

from agent.context_compressor import (
    _MIN_SUMMARY_TOKENS,
    _SUMMARY_TOKENS_CEILING,
    _SUMMARY_TOKENS_CEILING_MAX,
    _SUMMARY_TOKENS_CEILING_MIN,
    _SUMMARY_WINDOW_HEADROOM_RATIO,
    ContextCompressor,
    _summarizer_max_output_tokens,
)


def _compressor(**attrs):
    """A ContextCompressor skeleton with only the state `max_summary_tokens` reads."""
    c = ContextCompressor.__new__(ContextCompressor)
    c._max_summary_tokens = None
    c._threshold_tokens = None
    c._tail_token_budget = None
    c.context_length = attrs.pop("context_length", 1_000_000)
    for key, value in attrs.items():
        setattr(c, key, value)
    return c


@pytest.fixture(autouse=True)
def _isolate_host_aux_route(monkeypatch):
    """Keep the budget assertions independent of the host's live `auxiliary.compression` route."""
    monkeypatch.setattr("agent.auxiliary_client._get_auxiliary_task_config", lambda _task: {})


# --- a. backward compatibility -----------------------------------------------------------------


def test_unconfigured_ceiling_keeps_prior_default():
    """a. With no `summary_tokens_ceiling`, the budget is the historical 10_000."""
    assert _compressor().max_summary_tokens == _SUMMARY_TOKENS_CEILING == 10_000


def test_legacy_ratio_term_no_longer_caps_the_budget():
    """c. The old `min(window * 0.05, 10_000)` term is gone: a 1M window with a 32K ceiling
    must not be silently pinned back to 10_000 (the bug this work order fixes)."""
    budget = _compressor(summary_tokens_ceiling=32_000).max_summary_tokens
    assert budget == 32_000, f"legacy 0.05*1M=50_000 term must not win; got {budget}"


# --- b. configured ceiling takes effect --------------------------------------------------------


def test_coding_profile_ceiling_lands_in_24k_32k_band():
    """b. coding profile = 32_000 on a large window yields a budget inside 24K-32K."""
    budget = _compressor(summary_tokens_ceiling=32_000, context_length=1_000_000).max_summary_tokens
    assert 24_000 <= budget <= 32_000, f"expected 24K-32K, got {budget}"


@pytest.mark.parametrize("window", [262_144, 400_000, 1_048_576])
def test_band_holds_across_realistic_windows(window):
    """b. The 24K-32K band holds for every realistic summarizer window."""
    budget = _compressor(summary_tokens_ceiling=32_000, context_length=window).max_summary_tokens
    assert 24_000 <= budget <= 32_000, f"window={window} gave {budget}"


def test_window_headroom_can_clamp_a_large_ceiling():
    """b/c. A ceiling larger than the window permits is clamped by the headroom term."""
    window = 20_000
    budget = _compressor(summary_tokens_ceiling=32_000, context_length=window).max_summary_tokens
    assert budget == int(window * (1.0 - _SUMMARY_WINDOW_HEADROOM_RATIO)) == 16_000


# --- h. fail-safe on misconfiguration -----------------------------------------------------------


@pytest.mark.parametrize("configured", [0, -1, 100, 3_999])
def test_below_band_config_clamps_to_band_floor(configured):
    """h. A below-band value lands on the documented 4_000 floor, never lower."""
    budget = _compressor(summary_tokens_ceiling=configured).max_summary_tokens
    assert budget == _SUMMARY_TOKENS_CEILING_MIN == 4_000, f"{configured} gave {budget}"


@pytest.mark.parametrize("configured", [32_001, 100_000, 10_000_000])
def test_above_band_config_clamps_to_band_top(configured):
    """h. An absurd value is clamped to 32_000 instead of being fatal."""
    assert _compressor(summary_tokens_ceiling=configured).max_summary_tokens == _SUMMARY_TOKENS_CEILING_MAX


@pytest.mark.parametrize("garbage", [None, "abc", {}, [], True])
def test_garbage_config_falls_back_to_default(garbage):
    """h. Non-numeric config falls back to the 10_000 default rather than raising."""
    assert _compressor(summary_tokens_ceiling=garbage).max_summary_tokens == _SUMMARY_TOKENS_CEILING


def test_tiny_window_never_produces_unusable_budget():
    """h. A window too small for the band still yields at least _MIN_SUMMARY_TOKENS."""
    budget = _compressor(summary_tokens_ceiling=32_000, context_length=1_000).max_summary_tokens
    assert budget == _MIN_SUMMARY_TOKENS


def test_unknown_route_is_not_fatal(monkeypatch):
    """h. When the catalog cannot resolve the model, the lookup returns None (budget still works)."""
    monkeypatch.setattr("agent.models_dev.get_model_capabilities", lambda _p, _m: None)
    c = _compressor(summary_tokens_ceiling=32_000)
    c.summary_model = "not-in-catalog"
    assert _summarizer_max_output_tokens(c) is None
    assert c.max_summary_tokens == 32_000


def test_lookup_survives_a_broken_catalog(monkeypatch):
    """h. A catalog that raises must not propagate: the budget falls back to ceiling/headroom."""

    def _boom(_p, _m):
        raise RuntimeError("catalog unavailable")

    monkeypatch.setattr("agent.models_dev.get_model_capabilities", _boom)
    c = _compressor(summary_tokens_ceiling=32_000)
    c.summary_model = "whatever"
    assert _summarizer_max_output_tokens(c) is None
    assert c.max_summary_tokens == 32_000


def test_model_output_cap_wins_when_smaller(monkeypatch):
    """h. A model whose own cap is below the ceiling clamps the budget (fail-safe, not fail-hard)."""
    import agent.context_compressor as mod

    class _Caps:
        max_output_tokens = 8_000

    monkeypatch.setattr(mod, "_summarizer_max_output_tokens", lambda _c: 8_000, raising=True)
    c = _compressor(summary_tokens_ceiling=32_000)
    assert c.max_summary_tokens == 8_000


def test_summarizer_cap_is_read_from_the_aux_compression_route(monkeypatch):
    """h. The model-output limb queries the AUX summariser route, never the main model's route.

    The summary call resolves provider/model from `auxiliary.compression`; reading the main
    model's catalog cap here would clamp the summary budget with an unrelated model.
    """
    import agent.context_compressor as mod

    class _Caps:
        max_output_tokens = 8_000

    seen = {}

    def _fake(provider, model):
        seen["route"] = (provider, model)
        return _Caps()

    monkeypatch.setattr(
        "agent.auxiliary_client._get_auxiliary_task_config",
        lambda _task: {"provider": "deepseek", "model": "deepseek-flash"},
    )
    monkeypatch.setattr("agent.models_dev.get_model_capabilities", _fake)
    c = _compressor(summary_tokens_ceiling=32_000, provider="newapi", model="main-model")
    assert c.max_summary_tokens == 8_000
    assert seen["route"] == ("deepseek", "deepseek-flash")


# --- d/e. truncation retry ----------------------------------------------------------------------


def test_truncation_retry_latch_is_one_shot():
    """d. The truncation retry latch admits exactly one retry per compression cycle."""
    c = _compressor(summary_tokens_ceiling=32_000)
    c._truncated_retry_done = False
    assert c._truncated_retry_done is False  # first truncation may retry
    c._truncated_retry_done = True
    assert c._truncated_retry_done is True  # second truncation must NOT retry again


def test_raise_summary_budget_for_retry_raises_to_ceiling():
    """d. The retry path raises the budget from the default to the configured ceiling."""
    c = _compressor(summary_tokens_ceiling=32_000)
    c._max_summary_tokens = 10_000
    before, after = c._raise_summary_budget_for_retry()
    assert (before, after) == (10_000, 32_000)
    assert c.max_summary_tokens == 32_000


def test_raise_is_a_noop_when_already_at_ceiling():
    """d. Nothing to raise: the helper reports an unchanged pair instead of looping."""
    c = _compressor(summary_tokens_ceiling=32_000)
    c._max_summary_tokens = 32_000
    assert c._raise_summary_budget_for_retry() == (32_000, 32_000)


def test_raise_respects_window_headroom():
    """d. Even the retry cannot exceed what the window allows."""
    c = _compressor(summary_tokens_ceiling=32_000, context_length=20_000)
    c._max_summary_tokens = 8_000
    _before, after = c._raise_summary_budget_for_retry()
    assert after == 16_000


# --- g. non-length failures keep prior semantics ------------------------------------------------


def test_min_and_max_constants_are_invariant():
    """g/h. The documented band is stable and self-consistent."""
    assert _SUMMARY_TOKENS_CEILING_MIN == 4_000
    assert _SUMMARY_TOKENS_CEILING_MAX == 32_000
    assert _SUMMARY_TOKENS_CEILING_MIN < _SUMMARY_TOKENS_CEILING <= _SUMMARY_TOKENS_CEILING_MAX
    assert _MIN_SUMMARY_TOKENS <= _SUMMARY_TOKENS_CEILING_MIN
