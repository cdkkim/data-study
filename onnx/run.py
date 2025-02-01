import onnxruntime as ort
import numpy as np

# Load the model
session = ort.InferenceSession("model.onnx")

# Get input name
input_name = session.get_inputs()[0].name

# Create input data
input_data = np.random.randn(1, 10).astype(np.float32)

# Run inference
output = session.run(None, {input_name: input_data})

print("Output:", output)
