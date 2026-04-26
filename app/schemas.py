"""central pydantic schemas — every contract between modules lives here."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# pricing tool contracts
# ---------------------------------------------------------------------------

PropertyType = Literal["flat", "terraced", "semi", "detached"]
RiskProfile = Literal["low", "medium", "high"]
FloodZone = Literal["standard", "medium", "high"]
ClaimsBand = Literal["0", "1", "2", "3+"]
SecurityLevel = Literal["none", "standard_locks", "alarm_and_deadlocks"]
PropertyAgeBand = Literal["post_2000", "1945_to_2000", "1900_to_1945", "pre_1900"]
PolicyType = Literal["standard", "landlord", "high_value", "flood_zone_extension"]


class PricingRequest(BaseModel):
    """input to the pricing tool — all fields the mock table needs to quote."""

    property_type: PropertyType
    bedrooms: int = Field(ge=1, le=5, description="bedroom count, 1 to 5")
    risk_profile: RiskProfile
    flood_zone: FloodZone = "standard"
    claims_last_5_years: ClaimsBand = "0"
    security: SecurityLevel = "standard_locks"
    property_age_band: PropertyAgeBand = "1945_to_2000"
    policy_type: PolicyType = "standard"


class AppliedModifier(BaseModel):
    """one row in the modifier breakdown — kept so the agent can show its working."""

    name: str
    value: str
    multiplier: float


class PriceQuote(BaseModel):
    """output of the pricing tool — indicative only, never a binding quote."""

    base_annual_premium_gbp: float
    final_annual_premium_gbp: float
    currency: Literal["GBP"] = "GBP"
    modifiers_applied: list[AppliedModifier]
    is_indicative: bool = True
    disclaimer: str = (
        "indicative estimate only — not a binding quote, "
        "actual price subject to underwriting"
    )


# ---------------------------------------------------------------------------
# policy lookup tool contracts
# ---------------------------------------------------------------------------


class PolicyChunk(BaseModel):
    """one retrieved chunk from the vector store — what the agent cites from."""

    chunk_id: str
    text: str
    source_file: str
    section_title: str
    distance: float = Field(ge=0.0, description="cosine distance, smaller is closer")


class PolicyLookupResult(BaseModel):
    """wrapper around a list of chunks so the agent gets one object back."""

    query: str
    chunks: list[PolicyChunk]
    count: int

    @field_validator("count")
    @classmethod
    def _count_matches(cls, v: int, info):
        # sanity check — count must equal len(chunks), prevents silent drift
        chunks = info.data.get("chunks", [])
        if v != len(chunks):
            raise ValueError(f"count {v} does not match chunks length {len(chunks)}")
        return v


# ---------------------------------------------------------------------------
# guardrail contracts
# ---------------------------------------------------------------------------

GuardrailVerdict = Literal["allow", "block"]


class GuardrailResult(BaseModel):
    """output of every guardrail — a verdict plus the reasons behind it."""

    verdict: GuardrailVerdict
    reasons: list[str] = Field(default_factory=list)
    rule_hits: list[str] = Field(
        default_factory=list,
        description="names of regex rules or judge dimensions that fired",
    )


# ---------------------------------------------------------------------------
# api contracts (FastAPI in/out)
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    """body of POST /ask — one user question per call."""

    question: str = Field(min_length=1, max_length=2000)
    session_id: Optional[str] = None


class AskResponse(BaseModel):
    """body returned by POST /ask — the answer plus the trail behind it."""

    answer: str
    citations: list[PolicyChunk] = Field(default_factory=list)
    quote: Optional[PriceQuote] = None
    refused: bool = False
    refusal_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# agent state — passed between langgraph nodes
# ---------------------------------------------------------------------------


class AgentState(BaseModel):
    """the single state object the langgraph nodes read and write."""

    question: str
    intent: Optional[Literal["pricing", "policy", "both", "out_of_scope"]] = None
    pricing_request: Optional[PricingRequest] = None
    quote: Optional[PriceQuote] = None
    policy_chunks: list[PolicyChunk] = Field(default_factory=list)
    draft_answer: Optional[str] = None
    final_answer: Optional[str] = None
    input_guardrail: Optional[GuardrailResult] = None
    output_guardrail: Optional[GuardrailResult] = None
    refused: bool = False
    refusal_reason: Optional[str] = None
    retry_count: int = 0
