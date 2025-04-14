import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import pickle
from sklearn.preprocessing import LabelEncoder
import json


import onnx

# Load your data
df = pd.read_csv("student_data.csv")

# Define input features and target
feature_columns = ['math_score', 'english_score', 'bahasa_score', 'distance']
X = df[feature_columns]
y = df['accepted_school']

# Define custom weights
weights = np.sqrt([0.2, 0.2, 0.2, 0.4])
X_scaled = X * weights

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Train the MLP model
mlp = MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)

# Predict probabilities for a new student
new_student = np.array([[97, 91, 58, 443.45]])  # Example values
new_student_scaled = new_student * weights
probs = mlp.predict_proba(new_student_scaled)[0]


# Get top 3 recommendations
top_3_indices = probs.argsort()[-5:][::-1]
top_3_schools = label_encoder.inverse_transform(top_3_indices)

print(f"probs:{probs}")
print(f"probs.type:{type(probs)}")
print(f"top_3_indices:{top_3_indices}")

print("Top 3 Recommended Schools:")
for rank, school in enumerate(top_3_schools, 1):
    print(f"{rank}. {school} (Prob: {probs[top_3_indices[rank-1]]:.2f})")

# Convert to ONNX
initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]

options = {
    id(mlp): {
        'zipmap': False  # Disable seq(map(...)), get plain tensor(float)
    }
}
onnx_model = convert_sklearn(mlp, initial_types=initial_type, options=options)
with open("school_recommender_mlp.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

#label encode


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save it
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)
with open("label_encoder.json", "w") as f:
    json.dump(le.classes_.tolist(), f)