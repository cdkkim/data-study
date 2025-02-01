import onnxoptimizer

# Load model
model = onnx.load("model.onnx")

# Apply optimizations
passes = ["eliminate_identity", "fuse_bn_into_conv"]
optimized_model = onnxoptimizer.optimize(model, passes)

# Save optimized model
onnx.save(optimized_model, "optimized_model.onnx")
