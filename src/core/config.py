"""
Configuration management for Bangla OCR Pipeline.
"""
from dataclasses import dataclass
from typing import Literal
import torch
import os


@dataclass
class OCRConfig:
    """Configuration for Bangla OCR Pipeline."""
    
    # Model paths
    yolo_model_path: str = "models/yolo/best.pt"
    qwen_model_name: str = "swapnillo/Bangla-OCR-SFT"
    onnx_model_name: str = "swapnillo/Bangla-OCR-SFT-ONNX"  # ONNX version on HuggingFace
    
    # Model cache directory (for downloaded models)
    models_cache_dir: str = "models/cache"
    
    # Directories
    output_dir: str = "output"
    crops_dir: str = "output/crops"
    
    # YOLO detection parameters
    yolo_confidence: float = 0.5
    yolo_class_id: int = 0  # Class ID for word detection
    
    # Inference parameters
    batch_size_gpu: int = 8
    batch_size_cpu: int = 4
    max_new_tokens: int = 128
    
    # Backend selection (auto = GPU→PyTorch, CPU→ONNX)
    backend: Literal["auto", "pytorch", "onnx"] = "auto"
    
    # Device
    device: str = "auto"  # auto, cpu, cuda
    
    # Model offloading (free memory after inference)
    offload_models: bool = True
    
    def __post_init__(self):
        """Auto-detect device if set to auto."""
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.crops_dir, exist_ok=True)
        os.makedirs(self.models_cache_dir, exist_ok=True)
        
        # Set HuggingFace cache directory
        os.environ['HF_HOME'] = self.models_cache_dir
        os.environ['TRANSFORMERS_CACHE'] = os.path.join(self.models_cache_dir, 'transformers')
    
    def get_batch_size(self) -> int:
        """Get appropriate batch size based on device."""
        return self.batch_size_gpu if self.device == "cuda" else self.batch_size_cpu
    
    def should_use_onnx(self) -> bool:
        """Determine if ONNX backend should be used."""
        if self.backend == "onnx":
            return True
        elif self.backend == "pytorch":
            return False
        else:  # auto
            # Use ONNX for CPU, PyTorch for GPU
            return self.device == "cpu"
