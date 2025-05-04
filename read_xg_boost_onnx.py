import pickle
import onnxruntime as ort
import numpy as np

# Load ONNX model
session = ort.InferenceSession("school_recommender_xg_boost.onnx")

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
new_student = np.array([[108.5,5584,-295.0053463876927, 949.0899528009696,-110.45409496678181]])
#weights = np.sqrt([0.2, 0.2, 0.2, 0.2, 0.2, 0.4])
new_student_scaled = np.array(new_student, dtype=np.float32)

# Run inference
outputs = session.run(None, {input_name: new_student_scaled})

# Get probability vector
probs = outputs[1]  # Usually the 2nd output from skl2onnx is probabilities
probs_idx_0 = probs[0]
# Get top 3 class indices
# top3_indices = np.argsort(probs[0])[::-1][:3]
# print("Top 3 indices:", top3_indices)


# Optionally map indices to labels
# labels = session.get_outputs()[0].name  # or define manually
# print("Top 3 classes:", top3_indices)  # You can map these to school names
print(f"probs:{probs}")

with open("label_encoder_xg_boost.pkl", "rb") as f:
    label_encoder = pickle.load(f)

print(f"label_encoder:{label_encoder.classes_}")

# For each sample in the test set
for i_std in range(len(probs)):
    # Get top 3 indices and probabilities
    top3_indices_new_student = np.argsort(probs[i_std])[-3:][::-1]
    print(f"top3_indices_new_student:{top3_indices_new_student}")
    top3_probs_new_student = probs[i_std][top3_indices_new_student]

    for rank_new_student, (idx_new_student, prob_new_student) in enumerate(zip(top3_indices_new_student, top3_probs_new_student), start=1):
        school_id = key = label_encoder.classes_[idx_new_student]
        print(f"{rank_new_student} idx_new_student:{idx_new_student}, school_id:{school_id} (Prob: {prob_new_student:.2f})")



top_3_indices = probs_idx_0.argsort()[-3:][::-1]
# Get top 3 school names
top_3_schools = label_encoder.inverse_transform(top_3_indices)

print("Top 3 schools:", top_3_schools)
print("Top 3 Recommended Schools:")
for rank, school in enumerate(top_3_schools, 1):
    print(f"{rank}. School:{school} (Prob: {probs_idx_0[top_3_indices[rank-1]]:.2f})")