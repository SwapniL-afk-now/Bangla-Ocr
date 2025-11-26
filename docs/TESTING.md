# Bangla OCR Pipeline - Testing Guide

## Setup Complete

✅ All dependencies installed
✅ Model caching configured (`models/cache/`)
✅ Auto-offloading after inference enabled

## What You Need

### 1. YOLO Model
Place your trained YOLO word detection model here:
```
models/yolo/best.pt
```

### 2. Test Images
Already provided in `images/` directory:
- `image.jpg` (288 KB)
- `image (1).jpg` (28 KB)

## Running the Pipeline

### Quick Test (Recommended First)
```bash
python test_pipeline.py
```
This will:
1. Initialize the pipeline with CPU backend
2. Download Qwen2-VL model to `models/cache/` (first time only)
3. Process first image from `images/` directory
4. Show OCR results
5. Offload model from memory

### Full Processing
```bash
python scripts/run_ocr.py \
  --image images/ \
  --yolo_model models/yolo/best.pt \
  --model_name Qwen/Qwen2-VL-2B-Instruct
```

## Model Management Features

### ✅ Automatic Caching
- Models download to `models/cache/` directory
- Reused on subsequent runs (no re-download)
- HuggingFace cache configured automatically

### ✅ Auto-Offloading
- Models freed from memory after processing
- Reduces memory footprint
- Configurable via `OCRConfig.offload_models`

### ✅ CPU Optimized
- Uses PyTorch CPU mode (no GPU required)
- Float32 precision for compatibility
- Low memory mode during loading

## Expected Behavior

### First Run
```
Loading Qwen2-VL model: Qwen/Qwen2-VL-2B-Instruct on cpu...
Cache directory: models/cache
Downloading model... (this may take several minutes)
✓ Model loaded successfully on cpu
✓ Model size: ~4000.0 MB  # Approximate for 2B model
```

### Subsequent Runs
```
Loading Qwen2-VL model: Qwen/Qwen2-VL-2B-Instruct on cpu...
Cache directory: models/cache
✓ Using cached model from models/cache/transformers/...
✓ Model loaded successfully on cpu
```

### After Processing
```
Offloading models...
Offloading model from memory...
✓ Model offloaded
Done!
```

## Output Structure

```
output/
├── test/               # Test run output
│   ├── image.txt      # Recognized text
│   └── image.json     # Detailed results
└── crops/              # Cropped word images
    └── image/
        ├── image_word_0000.jpg
        ├── image_word_0001.jpg
        └── ...
```

## Important Notes

> [!NOTE]
> **ONNX Runtime**: Currently using PyTorch CPU backend. Converting Qwen2-VL to ONNX is complex and requires specialized tools. PyTorch CPU mode provides good performance for this use case.

> [!NOTE]
> **Model Size**: Qwen2-VL-2B model is ~4GB. First download will take time depending on internet speed. Model is cached for future use.

> [!NOTE]
> **YOLO Model Required**: You must provide your trained YOLO model for word detection. Place it in `models/yolo/` directory.

## Next Steps

1. **Place YOLO model** in `models/yolo/best.pt`
2. **Run test**: `python test_pipeline.py`
3. **Check output** in `output/test/` directory
4. **Process all images**: Use `scripts/run_ocr.py` for full pipeline

## Troubleshooting

### "No module named 'qwen_vl_utils'"
```bash
pip install qwen-vl-utils
```

### "YOLO model not found"
Place your trained YOLO model at `models/yolo/best.pt`

### Out of Memory
- Reduce batch size in config: `batch_size_cpu=2`
- Use smaller Qwen model (if available)
- Process images one at a time
