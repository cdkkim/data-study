import torch
import torch.nn as nn
import onnx

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)

# Create model and dummy input
model = SimpleModel()
dummy_input = torch.randn(1, 10)

# Export to ONNX
torch.onnx.export(model, dummy_input, "model.onnx", input_names=["input"], output_names=["output"])
