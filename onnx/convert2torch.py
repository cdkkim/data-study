from onnx2torch import convert
import torch

onnx_model = onnx.load("model.onnx")
torch_model = convert(onnx_model)

# Run inference
input_tensor = torch.randn(1, 10)
output = torch_model(input_tensor)
print(output)
