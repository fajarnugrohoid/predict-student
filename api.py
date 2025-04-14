from fastapi import FastAPI
import numpy as np
import onnxruntime

# Load ONNX model
onnx_model_path = "knn_model_fixed.onnx"
session = onnxruntime.InferenceSession(onnx_model_path)

# Initialize FastAPI
app = FastAPI()

@app.get("/")
def home():
    return {"message": "KNN ONNX API is running!"}

@app.post("/predict/")
def predict(data: dict):
    # Extract input data
    X_input = np.array(data["features"], dtype=np.float32).reshape(1, -1)

    # Run inference
    inputs = {session.get_inputs()[0].name: X_input}
    output = session.run(None, inputs)

    # Return predicted class
    print(f"X_input:{X_input}")
    print(f"inputs:{inputs}")
    print(f"output:{output}")
    print(f"output:{output[0]}")
    print(f"output:{output[0][0]}")
    return {"prediction": output[0][0]}