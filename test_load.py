import onnx
import onnxruntime as ort


print(f"ort:{ort.get_device()}")
model = onnx.load("knn_model_fixed.onnx")

check_model = onnx.checker.check_model(model)
print(f"model.checker:{check_model}")

print(f"model.graph.output:{model.graph.output}")


for o in model.graph.output:
    print(f"o.name:{o.name, o.type}")


print("Inputs:")
for i in model.graph.input:
    print(i.name)

print("\nOutputs:")
for o in model.graph.output:
    print(o.name)
