"""
Convert Qwen2-VL/Qwen3-VL model to ONNX format for CPU optimization.

This script uses the optimum library to convert the model properly.
"""
import os
import argparse
from pathlib import Path


def convert_to_onnx(
    model_name: str = "swapnillo/Bangla-OCR-SFT",
    output_dir: str = "models/qwen_onnx",
    hf_token: str = None
):
    """
    Convert Qwen model to ONNX format.
    
    Args:
        model_name: HuggingFace model name
        output_dir: Directory to save ONNX model
        hf_token: HuggingFace token (if model is private)
    """
    try:
        from optimum.exporters.onnx import main_export
        from transformers import AutoConfig
        
        print("="*70)
        print("QWEN MODEL → ONNX CONVERSION")
        print("="*70)
        print(f"Model: {model_name}")
        print(f"Output: {output_dir}")
        print()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set HF token if provided
        if hf_token:
            os.environ['HF_TOKEN'] = hf_token
            print("✓ HuggingFace token set")
        
        # Check model config
        print("Checking model configuration...")
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        print(f"✓ Model type: {config.model_type}")
        
        # Export to ONNX
        print("\nStarting ONNX export...")
        print("⚠ This may take 10-20 minutes for large models...")
        
        main_export(
            model_name_or_path=model_name,
            output=output_dir,
            task="image-to-text",  # Vision-language task
            opset=14,  # ONNX opset version
            device="cpu",  # Export for CPU
            fp16=False,  # Use FP32 for CPU
            optimize="O2",  # Optimization level
        )
        
        print("\n" + "="*70)
        print("✓ CONVERSION COMPLETE!")
        print("="*70)
        print(f"ONNX model saved to: {output_dir}")
        print()
        print("Next steps:")
        print("1. Test the ONNX model locally")
        print("2. Upload to HuggingFace (optional)")
        print()
        
        return True
        
    except ImportError:
        print("❌ Error: 'optimum' library not installed")
        print()
        print("Install with:")
        print("  pip install optimum[exporters,onnxruntime]")
        print()
        return False
        
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        print()
        print("Common issues:")
        print("1. Model architecture not supported by ONNX")
        print("2. Missing dependencies (install optimum, onnx, onnxruntime)")
        print("3. Insufficient memory (try on a machine with more RAM)")
        print()
        print("Alternative: Use PyTorch CPU mode (slower but works)")
        return False


def upload_to_huggingface(
    onnx_dir: str,
    repo_id: str,
    hf_token: str
):
    """
    Upload ONNX model to HuggingFace Hub.
    
    Args:
        onnx_dir: Directory containing ONNX model
        repo_id: HuggingFace repo ID (e.g., "username/model-name-onnx")
        hf_token: HuggingFace token
    """
    try:
        from huggingface_hub import HfApi, login
        
        print("="*70)
        print("UPLOADING TO HUGGINGFACE")
        print("="*70)
        print(f"Source: {onnx_dir}")
        print(f"Destination: {repo_id}")
        print()
        
        # Login
        login(token=hf_token)
        print("✓ Logged in to HuggingFace")
        
        # Initialize API
        api = HfApi()
        
        # Create repo
        api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")
        print(f"✓ Repository created/verified: {repo_id}")
        
        # Upload
        print("\nUploading files...")
        api.upload_folder(
            folder_path=onnx_dir,
            repo_id=repo_id,
            repo_type="model"
        )
        
        print("\n" + "="*70)
        print("✓ UPLOAD COMPLETE!")
        print("="*70)
        print(f"Model available at: https://huggingface.co/{repo_id}")
        print()
        
        return True
        
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert Qwen model to ONNX")
    parser.add_argument("--model", default="swapnillo/Bangla-OCR-SFT", help="HuggingFace model name")
    parser.add_argument("--output", default="models/qwen_onnx", help="Output directory")
    parser.add_argument("--hf_token", default=None, help="HuggingFace token")
    parser.add_argument("--upload", action="store_true", help="Upload to HuggingFace after conversion")
    parser.add_argument("--repo_id", default=None, help="HF repo ID for upload (e.g., username/model-onnx)")
    
    args = parser.parse_args()
    
    # Convert
    success = convert_to_onnx(args.model, args.output, args.hf_token)
    
    # Upload if requested
    if success and args.upload:
        if not args.repo_id:
            print("❌ Error: --repo_id required for upload")
            print("Example: --repo_id swapnillo/Bangla-OCR-SFT-ONNX")
            return
        
        if not args.hf_token:
            print("❌ Error: --hf_token required for upload")
            return
        
        upload_to_huggingface(args.output, args.repo_id, args.hf_token)


if __name__ == "__main__":
    main()
