
https://github.com/onnx/onnx

```bash
pip install torch torchvision onnx onnxruntime
```

### Steps

1. Convert a model to ONNX
```bash
python convert2onnx.py
```

2. Load and Inspect an ONNX model
```bash
python load.py
```

3. Run ONNX model with ONNX runtime
```bash
python run.py
```

4. Optimize ONNX models (optional)
```bash
pip install onnxoptimizer

python optimize.py
```

5. Convert ONNX model back to PyTorch (optional)
```bash
pip install onnx2torch

python convert2torch.py
```
