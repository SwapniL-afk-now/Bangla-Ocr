"""
Quick test script for Bangla OCR Pipeline.
Tests with images in the images/ directory.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.pipeline import BanglaOCRPipeline
from src.core.config import OCRConfig

def main():
    print("="*70)
    print("BANGLA OCR PIPELINE - QUICK TEST")
    print("="*70)
    
    # Check if YOLO model exists
    yolo_path = "models/yolo/best.pt"
    if not Path(yolo_path).exists():
        print(f"\n⚠ YOLO model not found at: {yolo_path}")
        print("Please place your YOLO model in models/yolo/ directory")
        print("\nFor this test, we'll skip detection and focus on OCR setup.")
        return
    
    # Create configuration
    config = OCRConfig(
        yolo_model_path=yolo_path,
        qwen_model_name="Qwen/Qwen2-VL-2B-Instruct",
        backend="auto",  # Will use PyTorch CPU if no GPU
        output_dir="output/test",
    )
    
    print(f"\nConfiguration:")
    print(f"  Device: {config.device}")
    print(f"  Backend: {config.backend}")
    print(f"  Batch size: {config.get_batch_size()}")
    print(f"  Models cache: {config.models_cache_dir}")
    
    try:
        # Initialize pipeline
        pipeline = BanglaOCRPipeline(config)
        
        # Find images
        images_dir = Path("images")
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        
        if not image_files:
            print("\n⚠ No images found in images/ directory")
            return
        
        print(f"\nFound {len(image_files)} images to process")
        
        # Process first image as test
        test_image = str(image_files[0])
        print(f"\nProcessing test image: {test_image}")
        
        result = pipeline.process_image(test_image, save_crops=True)
        
        # Save results
        output_file = Path("output/test") / f"{Path(test_image).stem}.txt"
        pipeline.save_results(result, str(output_file))
        
        print("\n" + "="*70)
        print("TEST COMPLETE!")
        print("="*70)
        print(f"Recognized text: {result['text'][:200] if result['text'] else '(no text)'}")
        print(f"Words detected: {result['num_words']}")
        print(f"Total time: {result['timings']['total']:.2f}s")
        print(f"Output: {output_file}")
        
        # Offload models
        print("\nOffloading models from memory...")
        pipeline.offload_models()
        print("✓ Done!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
