"""tests for the mock pricing tool — covers table reads, math, and error paths."""

from __future__ import annotations

import pytest

from app.schemas import PricingRequest
from app.tools.pricing_api import PricingError, _load_table, get_quote


def test_basic_quote_returns_correct_base() -> None:
    """semi 3-bed medium risk should match the table base of £420."""
    req = PricingRequest(
        property_type="semi",
        bedrooms=3,
        risk_profile="medium",
    )
    quote = get_quote(req)
    assert quote.base_annual_premium_gbp == 420.0


def test_modifier_math_is_multiplicative() -> None:
    """final = base * flood * claims * security * age * policy_type."""
    req = PricingRequest(
        property_type="semi",
        bedrooms=3,
        risk_profile="medium",
        flood_zone="medium",          # x1.35
        claims_last_5_years="1",      # x1.15
        security="alarm_and_deadlocks",  # x0.9
        property_age_band="1945_to_2000",  # x1.0
        policy_type="standard",       # x1.0
    )
    quote = get_quote(req)
    expected = round(420.0 * 1.35 * 1.15 * 0.9 * 1.0 * 1.0, 2)
    assert quote.final_annual_premium_gbp == expected


def test_quote_includes_all_five_modifiers() -> None:
    """every quote should report flood, claims, security, age, policy_type — five rows."""
    req = PricingRequest(
        property_type="flat", bedrooms=1, risk_profile="low"
    )
    quote = get_quote(req)
    names = [m.name for m in quote.modifiers_applied]
    assert names == [
        "flood_zone",
        "claims_last_5_years",
        "security",
        "property_age_band",
        "policy_type",
    ]


def test_quote_is_marked_indicative() -> None:
    """compliance check — every quote must be flagged indicative, never binding."""
    req = PricingRequest(
        property_type="detached", bedrooms=4, risk_profile="medium"
    )
    quote = get_quote(req)
    assert quote.is_indicative is True
    assert quote.currency == "GBP"
    # the disclaimer wording is the FCA-friendly bit — must mention "indicative"
    assert "indicative" in quote.disclaimer.lower()
    assert "binding" in quote.disclaimer.lower()


def test_high_flood_zone_uplifts_premium() -> None:
    """high flood zone is a 1.80x modifier — premium should jump by 80% over standard."""
    base_req = PricingRequest(
        property_type="terraced",
        bedrooms=2,
        risk_profile="low",
        flood_zone="standard",
    )
    high_req = PricingRequest(
        property_type="terraced",
        bedrooms=2,
        risk_profile="low",
        flood_zone="high",
    )
    base_quote = get_quote(base_req)
    high_quote = get_quote(high_req)
    # ratio of finals must equal the flood modifier ratio (1.80 / 1.00)
    ratio = high_quote.final_annual_premium_gbp / base_quote.final_annual_premium_gbp
    assert round(ratio, 2) == 1.80


def test_table_miss_raises_pricing_error() -> None:
    """no row in the table for these inputs — must raise PricingError, not return junk."""
    # 5-bed flat is not a row in mock_pricing_table.json
    req = PricingRequest(
        property_type="flat",
        bedrooms=5,
        risk_profile="high",
    )
    with pytest.raises(PricingError) as exc_info:
        get_quote(req)
    # error message must name the missing combination — helps debugging in prod
    msg = str(exc_info.value)
    assert "flat" in msg
    assert "5" in msg
    assert "high" in msg


def test_load_table_is_cached() -> None:
    """lru_cache(maxsize=1) means two calls return the same dict object — same id()."""
    first = _load_table()
    second = _load_table()
    # same object in memory — proves the cache is doing its job
    assert first is second
