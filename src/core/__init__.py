"""Bangla OCR Pipeline - Core modules."""

from .config import OCRConfig
from .word_detector import WordDetector, WordBox
from .pipeline import BanglaOCRPipeline

__all__ = [
    "OCRConfig",
    "WordDetector",
    "WordBox",
    "BanglaOCRPipeline",
]
