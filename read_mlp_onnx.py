import pickle
import onnxruntime as ort
import numpy as np

# Load ONNX model
session = ort.InferenceSession("school_recommender_mlp.onnx")

# Input name
input_name = session.get_inputs()[0].name
print(f"input_name:{input_name}")

# Print output names and shapes
for output in session.get_outputs():
    print("Output Name:", output.name)
    print("Shape:", output.shape)
    print("Type:", output.type)

# Input: one row with 4 features
#X = np.array([[97, 91, 58, 443.45]], dtype=np.float32)

new_student = np.array([[56,72,98, -6.987890173907045,107.76574571313512, 443.45]])
weights = np.sqrt([0.2, 0.2, 0.2, 0.2, 0.2, 0.4])
new_student_scaled = np.array(new_student * weights, dtype=np.float32)

# Run inference
outputs = session.run(None, {input_name: new_student_scaled})

# Get probability vector
probs = outputs[1]  # Usually the 2nd output from skl2onnx is probabilities
probs = probs[0]
# Get top 3 class indices
# top3_indices = np.argsort(probs[0])[::-1][:3]
# print("Top 3 indices:", top3_indices)


# Optionally map indices to labels
# labels = session.get_outputs()[0].name  # or define manually
# print("Top 3 classes:", top3_indices)  # You can map these to school names
print(f"probs:{probs}")

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

top_3_indices = probs.argsort()[-5:][::-1]
# Get top 3 school names
top_3_schools = label_encoder.inverse_transform(top_3_indices)

print("Top 3 schools:", top_3_schools)
print("Top 3 Recommended Schools:")
for rank, school in enumerate(top_3_schools, 1):
    print(f"{rank}. School:{school} (Prob: {probs[top_3_indices[rank-1]]:.2f})")