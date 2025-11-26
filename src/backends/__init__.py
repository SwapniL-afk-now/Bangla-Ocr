"""Bangla OCR Pipeline - Inference backends."""

from .qwen_ocr import QwenOCR
from .qwen_ocr_pytorch import QwenOCRPyTorch
from .qwen_ocr_onnx import QwenOCRONNX

__all__ = [
    "QwenOCR",
    "QwenOCRPyTorch",
    "QwenOCRONNX",
]
