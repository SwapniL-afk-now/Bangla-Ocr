"""
ONNX Runtime inference module (CPU backend).
"""
import onnxruntime as ort
import numpy as np
from typing import List
from PIL import Image
import cv2
import os


class QwenOCRONNX:
    """ONNX Runtime-based OCR inference for optimized CPU performance."""
    
    def __init__(self, onnx_model_path: str):
        """
        Initialize ONNX Runtime session.
        
        Args:
            onnx_model_path: Path to ONNX model directory or .onnx file
        """
        self.onnx_model_path = onnx_model_path
        
        # Find ONNX model file
        if os.path.isdir(onnx_model_path):
            # Look for .onnx file in directory
            onnx_files = [f for f in os.listdir(onnx_model_path) if f.endswith('.onnx')]
            if not onnx_files:
                raise ValueError(f"No .onnx file found in {onnx_model_path}")
            model_file = os.path.join(onnx_model_path, onnx_files[0])
        else:
            model_file = onnx_model_path
        
        print(f"Loading ONNX model: {model_file}...")
        
        # Create ONNX Runtime session with CPU optimizations
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = os.cpu_count()
        
        self.session = ort.InferenceSession(
            model_file,
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        
        print(f"ONNX model loaded successfully on CPU")
        print(f"Using {sess_options.intra_op_num_threads} CPU threads")
    
    def recognize_batch(self, images: List[np.ndarray], max_new_tokens: int = 128) -> List[str]:
        """
        Perform OCR on a batch of word images using ONNX.
        
        Args:
            images: List of numpy arrays (BGR format from OpenCV)
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            List of recognized text strings
        """
        if not images:
            return []
        
        # Convert images to RGB
        rgb_images = [cv2_to_rgb(img) for img in images]
        
        # Preprocess images
        processed_images = self._preprocess_batch(rgb_images)
        
        # Run inference
        results = []
        for processed_img in processed_images:
            # Prepare inputs
            inputs = {
                "pixel_values": processed_img,
            }
            
            # Run ONNX inference
            outputs = self.session.run(None, inputs)
            
            # Decode output
            # Note: This is a simplified version. Full implementation would need
            # proper tokenizer integration for decoding
            text = self._decode_output(outputs[0])
            results.append(text)
        
        return results
    
    def _preprocess_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Preprocess images for ONNX model.
        
        Args:
            images: List of RGB numpy arrays
            
        Returns:
            List of preprocessed image tensors
        """
        processed = []
        for img in images:
            # Resize to expected input size (depends on model)
            # This is a placeholder - actual preprocessing depends on model
            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize((224, 224))  # Placeholder size
            
            # Convert to tensor format
            img_array = np.array(pil_img).astype(np.float32) / 255.0
            img_array = np.transpose(img_array, (2, 0, 1))  # HWC to CHW
            img_array = np.expand_dims(img_array, 0)  # Add batch dimension
            
            processed.append(img_array)
        
        return processed
    
    def _decode_output(self, output: np.ndarray) -> str:
        """
        Decode model output to text.
        
        Note: This is a placeholder. Full implementation requires
        proper tokenizer integration from the Qwen model.
        
        Args:
            output: Model output tensor
            
        Returns:
            Decoded text string
        """
        # Placeholder implementation
        # In production, this would use the Qwen tokenizer to decode token IDs
        return ""


def cv2_to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert OpenCV BGR image to RGB."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
