"""
Main CLI script for running Bangla OCR.

Usage:
    # Process single image with auto backend selection
    python run_ocr.py --image test.jpg --yolo_model best.pt --model_name Qwen/Qwen2-VL-2B-Instruct
    
    # Process folder with PyTorch backend
    python run_ocr.py --image images/ --yolo_model best.pt --model_name Qwen/Qwen2-VL-2B-Instruct --backend pytorch
    
    # Use ONNX backend for CPU
    python run_ocr.py --image test.jpg --yolo_model best.pt --backend onnx --onnx_model models/qwen_onnx
"""
import argparse
import os
import sys
from pathlib import Path
from typing import List
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.pipeline import BanglaOCRPipeline
from src.core.config import OCRConfig


def get_image_files(path: str) -> List[str]:
    """
    Get list of image files from a path (file or directory).
    
    Args:
        path: Path to image file or directory
        
    Returns:
        List of image file paths
    """
    path_obj = Path(path)
    
    if path_obj.is_file():
        return [str(path_obj)]
    elif path_obj.is_dir():
        # Get all image files
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in extensions:
            image_files.extend(path_obj.glob(f'*{ext}'))
            image_files.extend(path_obj.glob(f'*{ext.upper()}'))
        return [str(f) for f in sorted(image_files)]
    else:
        raise ValueError(f"Path not found: {path}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Bangla OCR Pipeline - Extract Bangla text from images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single image with auto backend
  python run_ocr.py --image test.jpg --yolo_model best.pt --model_name Qwen/Qwen2-VL-2B-Instruct
  
  # Process folder with specific backend
  python run_ocr.py --image images/ --yolo_model best.pt --backend pytorch
  
  # Use ONNX for CPU optimization
  python run_ocr.py --image test.jpg --yolo_model best.pt --backend onnx --onnx_model models/qwen_onnx
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image file or directory"
    )
    parser.add_argument(
        "--yolo_model",
        type=str,
        required=True,
        help="Path to YOLO model weights file"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="Qwen model name from HuggingFace (default: Qwen/Qwen2-VL-2B-Instruct)"
    )
    parser.add_argument(
        "--onnx_model",
        type=str,
        default=None,
        help="Path to ONNX model (required if --backend=onnx)"
    )
    
    # Backend selection
    parser.add_argument(
        "--backend",
        type=str,
        choices=["auto", "pytorch", "onnx"],
        default="auto",
        help="Inference backend: auto (default), pytorch, or onnx"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Output directory for results (default: output)"
    )
    parser.add_argument(
        "--no_crops",
        action="store_true",
        help="Don't save cropped word images"
    )
    parser.add_argument(
        "--save_json",
        action="store_true",
        help="Save detailed results as JSON"
    )
    
    # Processing arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for OCR (auto-selected if not specified)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="YOLO confidence threshold (default: 0.5)"
    )
    
    args = parser.parse_args()
    
    # Validate ONNX backend
    if args.backend == "onnx" and not args.onnx_model:
        parser.error("--onnx_model is required when using --backend=onnx")
    
    # Create configuration
    config = OCRConfig(
        yolo_model_path=args.yolo_model,
        qwen_model_name=args.model_name,
        onnx_model_name=args.onnx_model if args.onnx_model else "swapnillo/Bangla-OCR-SFT-ONNX",
        output_dir=args.output_dir,
        yolo_confidence=args.confidence,
        batch_size_gpu=args.batch_size if args.batch_size else 8,
        batch_size_cpu=args.batch_size if args.batch_size else 4,
        backend=args.backend,
    )
    
    # Get image files
    print("Scanning for images...")
    image_files = get_image_files(args.image)
    
    # Initialize pipeline
    try:
        pipeline = BanglaOCRPipeline(config)
    except Exception as e:
        print(f"\n❌ Error initializing pipeline: {e}")
        if "ONNX model not found" in str(e):
            print("\nTo create ONNX model, run:")
            print("  python convert_to_onnx.py --model_name {} --output_dir {}".format(
                args.model_name, args.onnx_model or "models/qwen_onnx"
            ))
        return
    
    # Process images
    print("\n" + "="*70)
    print("Starting OCR Processing")
    print("="*70 + "\n")
    
    all_results = []
    total_start = time.time()
    
    for idx, image_path in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] Processing: {Path(image_path).name}")
        print("-" * 70)
        
        try:
            # Process image
            result = pipeline.process_image(
                image_path,
                save_crops=not args.no_crops
            )
            all_results.append(result)
            
            # Save text output
            image_name = Path(image_path).stem
            txt_path = os.path.join(config.output_dir, f"{image_name}.txt")
            pipeline.save_results(result, txt_path)
            
            # Save JSON if requested
            if args.save_json:
                json_path = os.path.join(config.output_dir, f"{image_name}.json")
                pipeline.save_results(result, json_path)
            
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}\n")
            continue
    
    # Summary
    total_time = time.time() - total_start
    print("\n" + "="*70)
    print("Processing Complete!")
    print("="*70)
    print(f"Total images processed: {len(all_results)}/{len(image_files)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per image: {total_time/max(len(all_results), 1):.2f}s")
    print(f"Backend used: {pipeline.ocr_engine.get_backend_name()}")
    print(f"Output directory: {config.output_dir}")
    print("="*70)
    
    # Offload models to free memory
    print("\nOffloading models...")
    pipeline.offload_models()
    print("Done!")


if __name__ == "__main__":
    main()
