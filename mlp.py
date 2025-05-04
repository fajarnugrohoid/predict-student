import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Helper function to transform lat/lng to 3D coordinates
def latlng_to_xyz(lat, lng):
    lat_rad = np.radians(lat)
    lng_rad = np.radians(lng)
    x = np.cos(lat_rad) * np.cos(lng_rad)
    y = np.cos(lat_rad) * np.sin(lng_rad)
    z = np.sin(lat_rad)
    #return x, y, z original_code
    return x*1000, y*1000, z*1000

# Load your data
df = pd.read_csv("students.csv")

# Transform latitude and longitude to xyz
#df['x'], df['y'], df['z'] = latlng_to_xyz(df['student_lat'], df['student_lng'])

# Define input features and target
# feature_columns = ['math_score', 'english_score', 'bahasa_score', 'student_latitude', 'student_longitude', 'distance']
feature_columns = ['age','score', 'x', 'y', 'z']
X = df[feature_columns]
y = df['accepted_school_id']

# Define custom weights
#weights = np.sqrt([0.1, 0.1, 0.1, 0.3, 0.3, 0.1])
#weights = np.sqrt([1, 1, 1, 1, 1])
#X_scaled = X * weights

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Train the MLP model
mlp = MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)


y_pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
recall = recall_score(y_test, y_pred, average='macro', zero_division=1)
f1 = f1_score(y_test, y_pred, average='macro')
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print("Confusion Matrix:")
print(conf_matrix)

# Predict probabilities for a new student

# new_student = np.array([[91,72,80,-6.961499698511773,107.75627091236132, 1055.45]])  # Example values math_score,english_score,bahasa_score,student_latitude,student_longitude,distance
new_student = np.array([[5639.0,411.129,-298.9524912572309,946.6040813158946,-120.69847226533716]])  # Example values
feature_names = ['age','score', 'x', 'y', 'z']  # <<< match your training columns
new_student_df = pd.DataFrame(new_student, columns=feature_names)

# Transform new student's lat/lng
#new_student_df['x'], new_student_df['y'], new_student_df['z'] = latlng_to_xyz(
#    new_student_df['student_lat'], new_student_df['student_lng']
#)
#new_student_ready = new_student_df[['age', 'score', 'x', 'y', 'z']]

#new_student_scaled = scaler.transform(new_student_ready * weights)
#new_student_scaled = scaler.transform(new_student_ready)
new_student_scaled = scaler.transform(new_student_df)
probs = mlp.predict_proba(new_student_scaled)[0]


# Get top 3 recommendations
top_3_indices = probs.argsort()[-3:][::-1]
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