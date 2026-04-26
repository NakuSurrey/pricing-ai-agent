"""mock pricing tool — reads the static table, applies modifiers, returns a quote."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from app.logging_config import logger
from app.schemas import (
    AppliedModifier,
    PriceQuote,
    PricingRequest,
)

# resolve the data file relative to the project root, not the working dir
PRICING_TABLE_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "mock_pricing_table.json"
)


class PricingError(Exception):
    """raised when the pricing table has no row for the given inputs."""


@lru_cache(maxsize=1)
def _load_table() -> dict:
    """load the json once, cache it — the table never changes at runtime."""
    if not PRICING_TABLE_PATH.exists():
        raise PricingError(f"pricing table not found at {PRICING_TABLE_PATH}")
    with PRICING_TABLE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_base_premium(
    table: dict,
    property_type: str,
    bedrooms: int,
    risk_profile: str,
) -> float:
    """walk the rules list, return the first matching base premium."""
    for row in table["base_rules"]:
        if (
            row["property_type"] == property_type
            and row["bedrooms"] == bedrooms
            and row["risk_profile"] == risk_profile
        ):
            return float(row["base_annual_premium"])
    raise PricingError(
        f"no base premium for {property_type=}, {bedrooms=}, {risk_profile=}"
    )


def get_quote(request: PricingRequest) -> PriceQuote:
    """
    main entry point — takes a validated PricingRequest, returns a PriceQuote.
    formula: final = base * flood_zone * claims * security * age * policy_type
    """
    table = _load_table()
    modifiers = table["modifiers"]

    base = _find_base_premium(
        table,
        request.property_type,
        request.bedrooms,
        request.risk_profile,
    )

    # pull each modifier value and build the breakdown row at the same time
    applied: list[AppliedModifier] = []

    flood_mult = modifiers["flood_zone"][request.flood_zone]
    applied.append(
        AppliedModifier(name="flood_zone", value=request.flood_zone, multiplier=flood_mult)
    )

    claims_mult = modifiers["claims_last_5_years"][request.claims_last_5_years]
    applied.append(
        AppliedModifier(
            name="claims_last_5_years",
            value=request.claims_last_5_years,
            multiplier=claims_mult,
        )
    )

    security_mult = modifiers["security"][request.security]
    applied.append(
        AppliedModifier(name="security", value=request.security, multiplier=security_mult)
    )

    age_mult = modifiers["property_age_band"][request.property_age_band]
    applied.append(
        AppliedModifier(
            name="property_age_band",
            value=request.property_age_band,
            multiplier=age_mult,
        )
    )

    policy_mult = modifiers["policy_type"][request.policy_type]
    applied.append(
        AppliedModifier(
            name="policy_type", value=request.policy_type, multiplier=policy_mult
        )
    )

    # multiplicative formula — same order as the table notes say
    final = base * flood_mult * claims_mult * security_mult * age_mult * policy_mult
    final_rounded = round(final, 2)

    logger.info(
        "pricing quote built — base={} final={} for {}",
        base,
        final_rounded,
        request.model_dump(),
    )

    return PriceQuote(
        base_annual_premium_gbp=base,
        final_annual_premium_gbp=final_rounded,
        modifiers_applied=applied,
    )


if __name__ == "__main__":
    # quick manual check — only runs when this file is executed directly
    sample = PricingRequest(
        property_type="semi",
        bedrooms=3,
        risk_profile="medium",
        flood_zone="medium",
        claims_last_5_years="1",
        security="alarm_and_deadlocks",
        property_age_band="1945_to_2000",
        policy_type="standard",
    )
    quote = get_quote(sample)
    print(quote.model_dump_json(indent=2))
