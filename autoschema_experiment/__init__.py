"""Isolated AutoSchemaKG-inspired experiment; no database access."""
from pathlib import Path
import sys
PIPELINE = Path(__file__).resolve().parents[1] / "pubmed_literature_extraction"
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))
VERSION = "0.1"
