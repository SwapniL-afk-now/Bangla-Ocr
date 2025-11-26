"""
Example usage of the Bangla OCR Pipeline.

This script demonstrates how to use the pipeline programmatically.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.pipeline import BanglaOCRPipeline
from src.core.config import OCRConfig
import os


def example_basic_usage():
    """Basic usage example."""
    print("=" * 70)
    print("Example 1: Basic Usage with Auto Backend")
    print("=" * 70)
    
    # Create configuration
    config = OCRConfig(
        yolo_model_path="path/to/your/yolo/model.pt",
        qwen_model_name="Qwen/Qwen2-VL-2B-Instruct",
        output_dir="output_example1",
        backend="auto"  # Automatically selects best backend
    )
    
    # Initialize pipeline
    pipeline = BanglaOCRPipeline(config)
    
    # Process an image
    result = pipeline.process_image(
        image_path="path/to/your/test_image.jpg",
        save_crops=True
    )
    
    # Print results
    print(f"\nRecognized text: {result['text']}")
    print(f"Number of words: {result['num_words']}")
    print(f"Processing time: {result['timings']['total']:.2f}s")
    print(f"Backend used: {result['backend']}")
    
    # Save results
    pipeline.save_results(result, "output_example1/result.txt")
    pipeline.save_results(result, "output_example1/result.json")


def example_gpu_backend():
    """Example using PyTorch GPU backend."""
    print("\n" + "=" * 70)
    print("Example 2: Force PyTorch GPU Backend")
    print("=" * 70)
    
    config = OCRConfig(
        yolo_model_path="path/to/your/yolo/model.pt",
        qwen_model_name="Qwen/Qwen2-VL-2B-Instruct",
        output_dir="output_example2",
        backend="pytorch",  # Force PyTorch backend
        device="cuda"  # Force CUDA
    )
    
    pipeline = BanglaOCRPipeline(config)
    
    # Process image
    result = pipeline.process_image("path/to/your/test_image.jpg")
    
    print(f"\nBackend: {result['backend']}")
    print(f"Text: {result['text']}")


def example_onnx_backend():
    """Example using ONNX CPU backend."""
    print("\n" + "=" * 70)
    print("Example 3: ONNX CPU Backend for Optimized Performance")
    print("=" * 70)
    
    config = OCRConfig(
        yolo_model_path="path/to/your/yolo/model.pt",
        qwen_model_name="Qwen/Qwen2-VL-2B-Instruct",
        onnx_model_path="models/qwen_onnx",  # Path to converted ONNX model
        output_dir="output_example3",
        backend="onnx",  # Force ONNX backend
        batch_size_cpu=4  # Optimize batch size for CPU
    )
    
    pipeline = BanglaOCRPipeline(config)
    
    # Process image
    result = pipeline.process_image("path/to/your/test_image.jpg")
    
    print(f"\nBackend: {result['backend']}")
    print(f"OCR time: {result['timings']['ocr']:.2f}s")


def example_batch_processing():
    """Example processing multiple images."""
    print("\n" + "=" * 70)
    print("Example 4: Batch Processing Multiple Images")
    print("=" * 70)
    
    config = OCRConfig(
        yolo_model_path="path/to/your/yolo/model.pt",
        qwen_model_name="Qwen/Qwen2-VL-2B-Instruct",
        output_dir="output_batch",
    )
    
    pipeline = BanglaOCRPipeline(config)
    
    # List of images to process
    image_paths = [
        "path/to/image1.jpg",
        "path/to/image2.jpg",
        "path/to/image3.jpg",
    ]
    
    # Process all images
    for i, image_path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}] Processing {image_path}...")
        
        result = pipeline.process_image(image_path)
        
        # Save individual results
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        pipeline.save_results(result, f"output_batch/{image_name}.txt")
        
        print(f"  Text: {result['text'][:100]}...")


def example_custom_parameters():
    """Example with custom parameters."""
    print("\n" + "=" * 70)
    print("Example 5: Custom Parameters")
    print("=" * 70)
    
    config = OCRConfig(
        yolo_model_path="path/to/your/yolo/model.pt",
        qwen_model_name="Qwen/Qwen2-VL-7B-Instruct",  # Larger model
        output_dir="output_custom",
        yolo_confidence=0.6,  # Higher confidence threshold
        batch_size_gpu=16,  # Larger batch size for GPU
        max_new_tokens=256,  # More tokens for longer text
        backend="auto"
    )
    
    pipeline = BanglaOCRPipeline(config)
    
    result = pipeline.process_image("path/to/your/test_image.jpg")
    
    print(f"\nConfiguration used:")
    print(f"  Model: {config.qwen_model_name}")
    print(f"  Confidence: {config.yolo_confidence}")
    print(f"  Batch size: {config.get_batch_size()}")
    print(f"  Backend: {result['backend']}")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("Bangla OCR Pipeline - Usage Examples")
    print("=" * 70)
    print("\nNOTE: Update the paths in this script before running!")
    print("  - YOLO model path")
    print("  - Test image paths")
    print("  - ONNX model path (for ONNX examples)")
    print("\n")
    
    # Uncomment the examples you want to run:
    
    # example_basic_usage()
    # example_gpu_backend()
    # example_onnx_backend()
    # example_batch_processing()
    # example_custom_parameters()
    
    print("\nTo run examples, uncomment the function calls in main()")


if __name__ == "__main__":
    main()
