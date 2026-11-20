"""Conservative classification of explicitly negated statistical assertions.

This module does not estimate treatment effects. It only distinguishes a
reported null/nonsignificant result from an ordinary explicit negation, using
the local trigger gap and a tightly attached estimate with its confidence
interval when one is present.
"""

from __future__ import annotations

import math
import re
from typing import Any


_NON_SIGNIFICANCE = re.compile(
    r"\b(?:not\s+(?:(?:statistically|clearly)\s+)?significant(?:ly)?|"
    r"no\s+(?:(?:statistical|statistically)\s+)?significant(?:ly)?|"
    r"without\s+statistical\s+significance|nonsignificant|non-significant)\b",
    re.IGNORECASE,
)
_NUMBER = r"[+\-−]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?"
_ESTIMATE = re.compile(
    rf"\A\s*[\(\[]?\s*(?P<measure>RR|OR|HR|SMD|MD)\s*(?:=|:)?\s*"
    rf"(?P<point>{_NUMBER})\s*[,;]?\s*\(?\s*"
    rf"(?:95\s*%\s*CI|CI\s*95\s*%)\s*[:=]?\s*[,;]?\s*\(?\s*"
    rf"(?P<lower>{_NUMBER})\s*(?:-|–|—|−|\bto\b)\s*"
    rf"(?P<upper>{_NUMBER})",
    re.IGNORECASE,
)
_ATTACHMENT_PREFIX = re.compile(r"\A[\s,:(\[]{0,16}")


def _parse_attached_estimate(
    clause: str, *, char_offset: int, right_argument_end: int,
) -> dict[str, Any] | None:
    if not isinstance(right_argument_end, int) or not 0 <= right_argument_end <= len(clause):
        return None
    tail = clause[right_argument_end:]
    prefix = _ATTACHMENT_PREFIX.match(tail)
    if prefix is None:
        return None
    local_start = prefix.end()
    match = _ESTIMATE.match(tail[local_start:])
    if match is None:
        return None
    # Only allow a confidence-interval close after the upper bound. This keeps
    # the parser tied to one compact estimate and prevents scanning onward to a
    # second, unrelated statistic in the same clause.
    if match.end() < len(tail[local_start:]) and tail[local_start + match.end()] not in ")]} ,;.!?\n":
        return None
    measure = match.group("measure").upper()
    try:
        point = float(match.group("point").replace("−", "-"))
        lower = float(match.group("lower").replace("−", "-"))
        upper = float(match.group("upper").replace("−", "-"))
    except ValueError:
        return None
    if not all(map(math.isfinite, (point, lower, upper))) or not lower <= point <= upper:
        return None
    null = 1.0 if measure in {"RR", "OR", "HR"} else 0.0
    if null == 1.0 and lower < 0.0:
        return None
    start = right_argument_end + local_start
    end = start + match.end()
    return {
        "measure": measure,
        "point": point,
        "lower": lower,
        "upper": upper,
        "null": null,
        "char_start": char_offset + start,
        "char_end": char_offset + end,
        "text": clause[start:end],
    }


def classify_opposition(
    clause: str,
    *,
    char_offset: int,
    negation: tuple[int, int],
    trigger: tuple[int, int],
    right_argument_end: int,
) -> dict[str, Any]:
    """Classify a scoped negation, retaining exact offsets for attached stats.

    `negation`, `trigger`, and `right_argument_end` are clause-local offsets.
    An ordinary explicit negation defaults to COUNTEREVIDENCE. A local
    non-significance phrase or an attached CI crossing its measure's null value
    returns INCONCLUSIVE. A CI excluding the null does not by itself prove a
    biological effect; it simply does not weaken the negation classification.
    """
    if (
        not isinstance(negation, tuple) or len(negation) != 2
        or not isinstance(trigger, tuple) or len(trigger) != 2
        or not (0 <= negation[0] < negation[1] <= trigger[0] < trigger[1] <= len(clause))
    ):
        return {"polarity": "COUNTEREVIDENCE", "reasons": ["explicit_negation"]}

    gap = clause[negation[0]:trigger[0]]
    reasons: list[str] = []
    if _NON_SIGNIFICANCE.search(gap):
        reasons.append("explicit_nonsignificance_language")

    estimate = _parse_attached_estimate(
        clause, char_offset=char_offset, right_argument_end=right_argument_end,
    )
    if estimate is not None:
        if estimate["lower"] <= estimate["null"] <= estimate["upper"]:
            reasons.append("confidence_interval_crosses_null")
        else:
            reasons.append("confidence_interval_excludes_null")

    inconclusive = (
        "explicit_nonsignificance_language" in reasons
        or "confidence_interval_crosses_null" in reasons
    )
    result: dict[str, Any] = {
        "polarity": "INCONCLUSIVE" if inconclusive else "COUNTEREVIDENCE",
        "reasons": reasons or ["explicit_negation"],
    }
    if estimate is not None:
        result["estimate"] = estimate
    return result
