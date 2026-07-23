"""
Historical Record Linkage Engine (ktt_linkage).
Robust probabilistic entity resolution for OCR data using Splink v4.
"""

from ktt_linkage.engine import RecordLinkageEngine
from ktt_linkage.config import get_splink_settings

__all__ = ["RecordLinkageEngine", "get_splink_settings"]