import onnx

# Load the model
model = onnx.load("model.onnx")

# Check if the model is well-formed
onnx.checker.check_model(model)

# Print model graph structure
print(onnx.helper.printable_graph(model.graph))
