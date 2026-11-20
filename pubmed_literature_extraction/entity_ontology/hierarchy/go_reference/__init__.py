"""Public read-only interface for querying GO from LiverKG."""
from .agent import ReferenceAgent
from .schema import Concept, ReferenceResult

__all__ = ["Concept", "ReferenceAgent", "ReferenceResult"]
