# Bangla OCR Pipeline

End-to-end Bangla OCR pipeline using YOLO word detection and Qwen2-VL vision language model with dual backend support (PyTorch GPU / ONNX CPU).

## Project Structure

```
bangla-ocr/
├── src/                    # Source code
│   ├── core/              # Core modules
│   │   ├── config.py      # Configuration management
│   │   ├── word_detector.py  # Word detection with YOLO
│   │   └── pipeline.py    # Main OCR pipeline
│   └── backends/          # Inference backends
│       ├── qwen_ocr.py           # Unified backend interface
│       ├── qwen_ocr_pytorch.py   # PyTorch GPU backend
│       └── qwen_ocr_onnx.py      # ONNX CPU backend
├── scripts/               # Executable scripts
│   ├── run_ocr.py        # Main CLI script
│   ├── convert_to_onnx.py  # ONNX conversion utility
│   ├── example_usage.py   # Usage examples
│   └── text_detection.py  # Original detection script
├── models/                # Model weights (YOLO, ONNX)
├── images/                # Input images for testing
├── output/                # OCR results and cropped words
├── docs/                  # Documentation
├── requirements.txt       # Python dependencies
└── README.md             # This file
```


## Features

- **Word Detection**: YOLO-based word detection with spatial sorting (reading order)
- **Dual Backend**: 
  - PyTorch backend for GPU acceleration
  - ONNX Runtime backend for CPU optimization (~2-3x faster than PyTorch CPU)
- **Batch Processing**: Efficient batch inference for faster processing
- **Flexible Input**: Single image or folder processing
- **CLI Interface**: Easy-to-use command-line interface

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For ONNX export (optional, if converting models yourself)
pip install optimum[exporters]
```

## Quick Start

### 1. Basic Usage (Auto Backend)

```bash
python scripts/run_ocr.py \
  --image images/input.jpg \
  --yolo_model models/yolo_model.pt \
  --model_name Qwen/Qwen2-VL-2B-Instruct
```

### 2. Process Multiple Images

```bash
python scripts/run_ocr.py \
  --image images/ \
  --yolo_model models/yolo_model.pt \
  --model_name Qwen/Qwen2-VL-2B-Instruct
```

### 3. Use ONNX Backend for CPU

```bash
# First, convert model to ONNX (one-time setup)
python scripts/convert_to_onnx.py \
  --model_name Qwen/Qwen2-VL-2B-Instruct \
  --output_dir models/qwen_onnx

# Run OCR with ONNX backend
python scripts/run_ocr.py \
  --image images/input.jpg \
  --yolo_model models/yolo_model.pt \
  --backend onnx \
  --onnx_model models/qwen_onnx
```

### 4. Force PyTorch GPU Backend

```bash
python scripts/run_ocr.py \
  --image images/input.jpg \
  --yolo_model models/yolo_model.pt \
  --backend pytorch
```

## Command Line Arguments

### Required
- `--image`: Path to input image or folder
- `--yolo_model`: Path to YOLO model weights

### Model Options
- `--model_name`: Qwen model name (default: Qwen/Qwen2-VL-2B-Instruct)
- `--onnx_model`: Path to ONNX model (required if --backend=onnx)

### Backend Options
- `--backend`: Choose backend: `auto` (default), `pytorch`, or `onnx`

### Output Options
- `--output_dir`: Output directory (default: output)
- `--no_crops`: Don't save cropped word images
- `--save_json`: Save detailed results as JSON

### Processing Options
- `--batch_size`: Batch size for OCR (auto-selected if not specified)
- `--confidence`: YOLO confidence threshold (default: 0.5)

## Architecture

```
Input Image
    ↓
[Word Detector (YOLO)]
    ↓
Sorted Word Crops (Reading Order)
    ↓
[Qwen2-VL OCR Engine]
├─→ PyTorch Backend (GPU)
└─→ ONNX Backend (CPU)
    ↓
Bangla Text Output
```

## Modules

- `src/core/config.py`: Configuration management
- `src/core/word_detector.py`: YOLO-based word detection with spatial sorting
- `src/backends/qwen_ocr_pytorch.py`: PyTorch GPU backend
- `src/backends/qwen_ocr_onnx.py`: ONNX CPU backend
- `src/backends/qwen_ocr.py`: Unified OCR interface with auto backend selection
- `src/core/pipeline.py`: Main pipeline orchestrator
- `scripts/convert_to_onnx.py`: ONNX model conversion utility
- `scripts/run_ocr.py`: CLI interface

## Performance Tips

- **GPU Available**: Use default `--backend auto` or `--backend pytorch`
- **CPU Only**: Convert to ONNX and use `--backend onnx` for ~2-3x speedup
- **Batch Size**: Larger batches are faster but use more memory
  - GPU: 8-16 (depends on VRAM)
  - CPU: 4-8 (depends on RAM)

## Output

- **Text files**: `output/<image_name>.txt` - Recognized Bangla text
- **JSON files**: `output/<image_name>.json` - Detailed results (if --save_json)
- **Cropped words**: `output/crops/<image_name>/` - Individual word images

## Converting to ONNX

For optimal CPU performance, convert the Qwen model to ONNX:

```bash
# Using optimum library (recommended)
pip install optimum[exporters]
optimum-cli export onnx \
  --model Qwen/Qwen2-VL-2B-Instruct \
  models/qwen_onnx

# Or use the provided script
python scripts/convert_to_onnx.py \
  --model_name Qwen/Qwen2-VL-2B-Instruct \
  --output_dir models/qwen_onnx
```

## License

This project uses:
- Ultralytics YOLO (AGPL-3.0)
- Qwen2-VL (Apache-2.0)
- Other dependencies as per their respective licenses
