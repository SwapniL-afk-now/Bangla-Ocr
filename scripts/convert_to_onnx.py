"""
Convert Qwen2-VL PyTorch model to ONNX format for CPU optimization.
"""
import torch
import argparse
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from pathlib import Path
import os


def convert_to_onnx(
    model_name: str,
    output_dir: str,
    opset_version: int = 14,
    validate: bool = True
):
    """
    Convert Qwen2-VL model to ONNX format.
    
    Args:
        model_name: HuggingFace model name
        output_dir: Directory to save ONNX model
        opset_version: ONNX opset version
        validate: Whether to validate the conversion
    """
    print(f"Converting {model_name} to ONNX format...")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print("\n[1/4] Loading PyTorch model...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # ONNX works better with float32
        device_map="cpu"
    )
    processor = AutoProcessor.from_pretrained(model_name)
    
    model.eval()
    print("Model loaded successfully")
    
    # Prepare dummy inputs
    print("\n[2/4] Preparing dummy inputs...")
    dummy_input_ids = torch.randint(0, 1000, (1, 50), dtype=torch.long)
    dummy_attention_mask = torch.ones(1, 50, dtype=torch.long)
    dummy_pixel_values = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    
    dummy_inputs = {
        "input_ids": dummy_input_ids,
        "attention_mask": dummy_attention_mask,
        # Note: Actual Qwen2-VL inputs may differ
    }
    
    # Export to ONNX
    print("\n[3/4] Exporting to ONNX...")
    onnx_path = os.path.join(output_dir, "model.onnx")
    
    try:
        torch.onnx.export(
            model,
            tuple(dummy_inputs.values()),
            onnx_path,
            input_names=list(dummy_inputs.keys()),
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"}
            },
            opset_version=opset_version,
            do_constant_folding=True,
        )
        print(f"ONNX model saved to: {onnx_path}")
    except Exception as e:
        print(f"\n⚠ WARNING: Full ONNX export failed: {e}")
        print("\nThis is expected for Qwen2-VL models due to their complexity.")
        print("Alternative approach: Use optimum library for better ONNX support.")
        print("\nTry: pip install optimum[exporters]")
        print("Then: optimum-cli export onnx --model {model_name} {output_dir}")
        return False
    
    # Validate conversion
    if validate:
        print("\n[4/4] Validating conversion...")
        try:
            import onnx
            import onnxruntime as ort
            
            # Load and check ONNX model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print("✓ ONNX model is valid")
            
            # Test inference
            session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            print("✓ ONNX Runtime can load the model")
            
        except Exception as e:
            print(f"⚠ Validation warning: {e}")
    
    print("\n" + "="*50)
    print("Conversion completed!")
    print(f"ONNX model location: {onnx_path}")
    print("="*50)
    
    return True


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert Qwen2-VL PyTorch model to ONNX format"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="HuggingFace model name (default: Qwen/Qwen2-VL-2B-Instruct)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/qwen_onnx",
        help="Output directory for ONNX model (default: models/qwen_onnx)"
    )
    parser.add_argument(
        "--opset_version",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)"
    )
    parser.add_argument(
        "--no_validate",
        action="store_true",
        help="Skip validation step"
    )
    
    args = parser.parse_args()
    
    # Print alternative method notice
    print("\n" + "="*50)
    print("ONNX Conversion Tool for Qwen2-VL")
    print("="*50)
    print("\nNOTE: Qwen2-VL models are complex and may require the 'optimum' library")
    print("for proper ONNX export. If this script fails, please use:")
    print("\n  pip install optimum[exporters]")
    print(f"  optimum-cli export onnx --model {args.model_name} {args.output_dir}")
    print("\n" + "="*50 + "\n")
    
    # Attempt conversion
    success = convert_to_onnx(
        model_name=args.model_name,
        output_dir=args.output_dir,
        opset_version=args.opset_version,
        validate=not args.no_validate
    )
    
    if not success:
        print("\nPlease use the optimum library method shown above.")


if __name__ == "__main__":
    main()
