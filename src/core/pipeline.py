"""
Main Bangla OCR Pipeline orchestrator.
"""
import os
import time
from typing import List, Dict, Tuple
from pathlib import Path
import json

from src.core.word_detector import WordDetector, WordBox
from src.backends.qwen_ocr import QwenOCR
from src.core.config import OCRConfig


class BanglaOCRPipeline:
    """End-to-end Bangla OCR pipeline."""
    
    def __init__(self, config: OCRConfig):
        """
        Initialize the OCR pipeline.
        
        Args:
            config: OCRConfig object with all parameters
        """
        self.config = config
        
        # Initialize word detector
        print("Initializing Word Detector...")
        self.word_detector = WordDetector(
            model_path=config.yolo_model_path,
            confidence_threshold=config.yolo_confidence,
            target_class_id=config.yolo_class_id
        )
        
        # Initialize OCR engine
        print("\nInitializing OCR Engine...")
        self.ocr_engine = QwenOCR(
            model_name=config.qwen_model_name,
            backend=config.backend,
            onnx_model_name=config.onnx_model_name,
            device=config.device,
            cache_dir=config.models_cache_dir
        )
        
        print("\n" + "="*50)
        print("Pipeline initialized successfully!")
        print(f"Backend: {self.ocr_engine.get_backend_name()}")
        print(f"Batch size: {config.get_batch_size()}")
        print("="*50 + "\n")
    
    def process_image(self, image_path: str, save_crops: bool = True) -> Dict:
        """
        Process a single image and extract Bangla text.
        
        Args:
            image_path: Path to input image
            save_crops: Whether to save cropped word images
            
        Returns:
            Dictionary with results and metadata
        """
        start_time = time.time()
        
        # Get image name
        image_name = Path(image_path).stem
        
        print(f"Processing: {image_path}")
        
        # Step 1: Detect and crop words
        detect_start = time.time()
        word_boxes = self.word_detector.detect_and_crop(image_path)
        detect_time = time.time() - detect_start
        
        print(f"  ✓ Detected {len(word_boxes)} words in {detect_time:.2f}s")
        
        if not word_boxes:
            print("  ⚠ No words detected!")
            return {
                "image_path": image_path,
                "num_words": 0,
                "text": "",
                "words": [],
                "timings": {
                    "detection": detect_time,
                    "ocr": 0,
                    "total": time.time() - start_time
                },
                "backend": self.ocr_engine.get_backend_name()
            }
        
        # Save crops if requested
        if save_crops:
            crops_dir = os.path.join(self.config.crops_dir, image_name)
            self.word_detector.save_crops(word_boxes, crops_dir, image_name)
            print(f"  ✓ Saved crops to: {crops_dir}")
        
        # Step 2: Perform OCR in batches
        ocr_start = time.time()
        recognized_texts = self._recognize_batch(word_boxes)
        ocr_time = time.time() - ocr_start
        
        print(f"  ✓ OCR completed in {ocr_time:.2f}s")
        
        # Combine results
        total_time = time.time() - start_time
        
        # Create word-level results
        words_data = []
        for box, text in zip(word_boxes, recognized_texts):
            words_data.append({
                "index": box.index,
                "text": text,
                "bbox": [box.x1, box.y1, box.x2, box.y2],
                "confidence": box.confidence
            })
        
        # Combine all text
        full_text = " ".join(recognized_texts)
        
        result = {
            "image_path": image_path,
            "num_words": len(word_boxes),
            "text": full_text,
            "words": words_data,
            "timings": {
                "detection": detect_time,
                "ocr": ocr_time,
                "total": total_time
            },
            "backend": self.ocr_engine.get_backend_name()
        }
        
        print(f"  ✓ Total time: {total_time:.2f}s")
        print(f"  ✓ Text: {full_text[:100]}..." if len(full_text) > 100 else f"  ✓ Text: {full_text}")
        print()
        
        return result
    
    def _recognize_batch(self, word_boxes: List[WordBox]) -> List[str]:
        """
        Recognize text from word boxes using batch processing.
        
        Args:
            word_boxes: List of WordBox objects
            
        Returns:
            List of recognized text strings
        """
        batch_size = self.config.get_batch_size()
        all_texts = []
        
        # Process in batches
        for i in range(0, len(word_boxes), batch_size):
            batch = word_boxes[i:i + batch_size]
            images = [box.image for box in batch]
            
            # Recognize batch
            texts = self.ocr_engine.recognize_batch(
                images,
                max_new_tokens=self.config.max_new_tokens
            )
            all_texts.extend(texts)
        
        return all_texts
    
    def save_results(self, result: Dict, output_path: str):
        """
        Save OCR results to file.
        
        Args:
            result: Result dictionary from process_image
            output_path: Path to save results (supports .txt and .json)
        """
        ext = Path(output_path).suffix.lower()
        
        if ext == '.json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        elif ext == '.txt':
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result['text'])
        else:
            raise ValueError(f"Unsupported output format: {ext}. Use .txt or .json")
        
        print(f"Results saved to: {output_path}")
    
    def offload_models(self):
        """Offload models from memory to free resources."""
        if self.config.offload_models:
            self.ocr_engine.offload()
