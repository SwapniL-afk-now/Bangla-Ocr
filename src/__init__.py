"""
Bangla OCR Pipeline

End-to-end Bangla OCR with YOLO detection and Qwen2-VL inference.
"""

from .core import OCRConfig, WordDetector, WordBox, BanglaOCRPipeline
from .backends import QwenOCR, QwenOCRPyTorch, QwenOCRONNX

__version__ = "1.0.0"

__all__ = [
    "OCRConfig",
    "WordDetector",
    "WordBox",
    "BanglaOCRPipeline",
    "QwenOCR",
    "QwenOCRPyTorch",
    "QwenOCRONNX",
]
