"""Provider failure categories shared by experiment harnesses."""

from __future__ import annotations


AUTH_ERROR = "AUTH_ERROR"
QUOTA_OR_TOKEN_EXHAUSTED = "QUOTA_OR_TOKEN_EXHAUSTED"
RATE_LIMITED = "RATE_LIMITED"
TRANSIENT_PROVIDER_ERROR = "TRANSIENT_PROVIDER_ERROR"
INVALID_RESPONSE = "INVALID_RESPONSE"
UNKNOWN_PROVIDER_ERROR = "UNKNOWN_PROVIDER_ERROR"


def classify_provider_error(error: str) -> str:
    text = str(error or "").casefold()
    if any(token in text for token in (
        "401", "403", "invalid token", "invalid api key", "unauthorized", "authentication",
    )):
        return AUTH_ERROR
    if any(token in text for token in (
        "quota", "insufficient", "balance", "credit", "token limit", "402",
    )):
        return QUOTA_OR_TOKEN_EXHAUSTED
    if "429" in text or "rate limit" in text or "too many requests" in text:
        return RATE_LIMITED
    if any(token in text for token in (
        "408", "425", "500", "502", "503", "504", "timeout", "timed out",
        "read timeout", "connect timeout", "connection",
        "bad gateway", "service unavailable", "temporar",
    )):
        return TRANSIENT_PROVIDER_ERROR
    if "json" in text or "decode" in text:
        return INVALID_RESPONSE
    return UNKNOWN_PROVIDER_ERROR


def is_retryable_provider_error(error: str) -> bool:
    return classify_provider_error(error) in {
        RATE_LIMITED,
        TRANSIENT_PROVIDER_ERROR,
    }
