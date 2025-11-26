"""
Unified OCR interface with automatic backend selection.
"""
from typing import List, Literal
import numpy as np
import torch
import os


class QwenOCR:
    """Factory class for Qwen OCR with automatic backend selection."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        backend: Literal["auto", "pytorch", "onnx"] = "auto",
        onnx_model_path: str = None,
        device: str = "auto",
        cache_dir: str = None
    ):
        """
        Initialize Qwen OCR with appropriate backend.
        
        Args:
            model_name: HuggingFace model name
            backend: Backend to use (auto/pytorch/onnx)
            onnx_model_path: Path to ONNX model (required if backend=onnx)
            device: Device to use (auto/cpu/cuda)
            cache_dir: Directory to cache downloaded models
        """
        self.model_name = model_name
        self.backend_choice = backend
        self.cache_dir = cache_dir
        
        # Auto-detect device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Determine backend
        if backend == "auto":
            # Use PyTorch for now (ONNX for Qwen2-VL is complex)
            use_onnx = False
        elif backend == "onnx":
            use_onnx = True
        else:  # pytorch
            use_onnx = False
        
        # Initialize appropriate backend
        if use_onnx:
            if not onnx_model_path:
                raise ValueError("onnx_model_path must be provided when using ONNX backend")
            
            if not os.path.exists(onnx_model_path):
                raise ValueError(
                    f"ONNX model not found at {onnx_model_path}. "
                    f"Please run convert_to_onnx.py first to create the ONNX model."
                )
            
            from src.backends.qwen_ocr_onnx import QwenOCRONNX
            self.backend = QwenOCRONNX(onnx_model_path)
            self.backend_name = "ONNX (CPU-optimized)"
        else:
            from src.backends.qwen_ocr_pytorch import QwenOCRPyTorch
            self.backend = QwenOCRPyTorch(model_name, device, cache_dir=cache_dir)
            self.backend_name = f"PyTorch ({device.upper()})"
        
        print(f"Using backend: {self.backend_name}")
    
    def recognize_batch(self, images: List[np.ndarray], max_new_tokens: int = 128) -> List[str]:
        """
        Perform OCR on a batch of word images.
        
        Args:
            images: List of numpy arrays (BGR format from OpenCV)
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            List of recognized text strings
        """
        return self.backend.recognize_batch(images, max_new_tokens)
    
    def offload(self):
        """Offload model from memory."""
        if hasattr(self.backend, 'offload'):
            self.backend.offload()
    
    def get_backend_name(self) -> str:
        """Get the name of the active backend."""
        return self.backend_name
