import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from haversine import haversine, Unit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
import json
import skl2onnx
from skl2onnx.common.data_types import FloatTensorType

weights = np.array([0.2, 0.2, 0.2, 0.4])
# Define the weighted Euclidean distance
def weighted_euclidean(x, y):
    diff = (x - y) * np.sqrt(weights)
    return np.linalg.norm(diff)

# Load dataset
df = pd.read_csv('student_data.csv')

# Define features and target
feature_columns = ['math_score', 'english_score', 'bahasa_score', 'distance']

X = df[feature_columns]  # Features
y = df['accepted_school']  # Target (school names)



# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

weights = np.sqrt([0.2, 0.2, 0.2, 0.4])

# Scale features manually
X_train_scaled = X_train * weights
X_test_scaled = X_test * weights

# Train KNN model (set k=5, but we will extract top 3 neighbors)
k = 5
knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
print(knn)
knn.fit(X_train_scaled, y_train)

# Plot the feature weights
plt.figure(figsize=(8, 5))
plt.barh(feature_columns, weights, color="skyblue")
plt.xlabel("Feature Weight")
plt.ylabel("Features")
plt.title("Feature Weight Distribution")
plt.show()

# Make predictions
y_pred = knn.predict(X_test_scaled)

# Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Example new student data
distance_new_student = haversine( (-6.932284674339599,107.78140745962706), (-6.928319,107.779412), unit=Unit.METERS)
print(f"distance_new_student:{distance_new_student}")
new_student = [[97,91,58, distance_new_student]]

# Get the indices of the 3 nearest neighbors
distances, indices = knn.kneighbors(new_student, n_neighbors=3)

# Get the schools of the 3 nearest neighbors
nearest_schools = y_train.iloc[indices[0]]

# Count occurrences of each school
school_counts = nearest_schools.value_counts()

# Get the top 3 recommended schools
top_3_schools = school_counts.index.tolist()

# Print results
print("Top 3 Recommended Schools:")
for rank, school in enumerate(top_3_schools, 1):
    print(f"{rank}. {school} (Neighbors: {school_counts[school]})")

# Save the model as JSON
model_data = {
    "n_neighbors": knn.n_neighbors,
    "X_train": X_train.values.tolist(),
    "y_train": y_train.values.tolist()
}

with open("knn_model.json", "w") as f:
    json.dump(model_data, f)

# Fix ONNX input shape issue (Set batch size to None)
initial_type = [("float_input", FloatTensorType([None, X_train.shape[1]]))]  # Dynamic batch size

options = {
    id(knn): {
        "output_class_labels": True,  # ⛔ don't include label strings
        "zipmap": False                # ⛔ don't wrap output in a map
    }
}

# Convert and save ONNX model
onnx_model = skl2onnx.convert_sklearn(knn, initial_types=initial_type,options=options)
with open("knn_model_fixed.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())