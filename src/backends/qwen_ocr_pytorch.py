"""
PyTorch-based Qwen2-VL/Qwen3-VL inference module (GPU backend).
"""
import torch
from transformers import AutoProcessor
try:
    from transformers import Qwen3VLForConditionalGeneration
    HAS_QWEN3 = True
except ImportError:
    from transformers import Qwen2VLForConditionalGeneration
    HAS_QWEN3 = False

from PIL import Image
import numpy as np
from typing import List, Union
import gc
import os


class QwenOCRPyTorch:
    """PyTorch-based OCR inference using Qwen2-VL/Qwen3-VL for GPU/CPU."""
    
    def __init__(self, model_name: str, device: str = "cuda", cache_dir: str = None):
        """
        Initialize PyTorch Qwen model.
        
        Args:
            model_name: HuggingFace model name (e.g., "swapnillo/Bangla-OCR-SFT")
            device: Device to use (cuda or cpu)
            cache_dir: Directory to cache downloaded models
        """
        self.device = device
        self.model_name = model_name
        self.cache_dir = cache_dir
        
        print(f"Loading Qwen model: {model_name} on {device}...")
        print(f"Cache directory: {cache_dir or 'default'}")
        
        # Check if model is already cached
        if cache_dir and os.path.exists(cache_dir):
            cached_model_path = os.path.join(cache_dir, 'transformers', model_name.replace('/', '--'))
            if os.path.exists(cached_model_path):
                print(f"✓ Using cached model from {cached_model_path}")
        
        # Load model and processor
        try:
            # Try loading with Qwen3VL if available, otherwise Qwen2VL
            model_class = Qwen3VLForConditionalGeneration if HAS_QWEN3 else Qwen2VLForConditionalGeneration
            print(f"Using model class: {model_class.__name__}")
            
            self.model = model_class.from_pretrained(
                model_name,
                torch_dtype=torch.float32,  # Use float32 for CPU compatibility
                device_map=device if device == "cuda" else "cpu",
                cache_dir=cache_dir,
                low_cpu_mem_usage=True,
                trust_remote_code=True  # Required for some new models
            )
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"\n⚠ Error loading model: {e}")
            print(f"Note: Using Qwen-VL models requires qwen-vl-utils package.")
            raise e
        
        self.model.eval()
        print(f"✓ Model loaded successfully on {device}")
        print(f"✓ Model size: ~{self._get_model_size_mb():.1f} MB")
    
    def _get_model_size_mb(self) -> float:
        """Get approximate model size in MB."""
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        return (param_size + buffer_size) / (1024 ** 2)
    
    def recognize_batch(self, images: List[np.ndarray], max_new_tokens: int = 128) -> List[str]:
        """
        Perform OCR on a batch of word images.
        
        Args:
            images: List of numpy arrays (BGR format from OpenCV)
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            List of recognized text strings
        """
        if not images:
            return []
        
        # Convert BGR to RGB PIL images
        pil_images = [Image.fromarray(cv2_to_rgb(img)) for img in images]
        
        # Prepare batch inputs
        texts = []
        image_inputs = []
        video_inputs = []
        
        # Specific prompt from kaggle.py
        prompt = "Transcribe all the handwritten Bangla text from the image. Respond only with the transcribed text. There are ultiple bangla text, you need to got hrough each line."
        
        for pil_img in pil_images:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "image": pil_img}
                    ]
                }
            ]
            
            # Prepare text input
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            texts.append(text)
            
            # Process vision info
            try:
                from qwen_vl_utils import process_vision_info
                img_in, vid_in = process_vision_info(messages)
                image_inputs.extend(img_in)
                if vid_in:
                    video_inputs.extend(vid_in)
            except ImportError:
                # Fallback if utils not available
                image_inputs.append(pil_img)
        
        # Batch tokenize and pad
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs if video_inputs else None,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        
        # Generate for the whole batch
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False  # Deterministic generation
            )
        
        # Trim generated IDs (skip input tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # Batch decode
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        # Clean up
        del inputs, generated_ids, generated_ids_trimmed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return [text.strip() for text in output_texts]
                    
            except Exception as e:
                print(f"⚠ Error processing image {idx}: {e}")
                results.append("")
        
        return results
    
    def offload(self):
        """Offload model from memory."""
        print("Offloading model from memory...")
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("✓ Model offloaded")
    
    def __del__(self):
        """Cleanup resources."""
        try:
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'processor'):
                del self.processor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass


def cv2_to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert OpenCV BGR image to RGB."""
    import cv2
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
