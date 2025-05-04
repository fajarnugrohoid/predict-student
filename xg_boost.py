from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
import pickle
import json



# Load data
df = pd.read_csv("students.csv")

# Features and raw target
original_features = ['score', 'age', 'x', 'y', 'z']

# Rename features to f0, f1, ..., fN
renamed_features = [f"f{i}" for i in range(len(original_features))]
feature_name_map = dict(zip(renamed_features, original_features))

# Replace column names
X = df[original_features].copy()
X.columns = renamed_features
y_raw = df['accepted_school_id']


# Split first
X_train, X_test, y_train_raw, y_test_raw = train_test_split(X, y_raw, test_size=0.2, random_state=42)

# Fit encoder ONLY on training labels
le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)

# Create a reverse map to decode predictions
index_to_id = dict(enumerate(le.classes_))

# Remove test samples with unseen labels
mask = y_test_raw.isin(le.classes_)
X_test_filtered = X_test[mask]
y_test_filtered_raw = y_test_raw[mask]

# Train model
model = xgb.XGBClassifier(eval_metric='mlogloss')
model.fit(X_train, y_train)

'''
new_student = {
    "score": 108.5,
    "age": 5584,
    "x": -295.0053463876927,
    "y": 949.0899528009696,
    "z": -110.45409496678181
} '''
new_student_renamed = {
    "f0": 108.5,
    "f1": 5584,
    "f2": -295.0053463876927,
    "f3": 949.0899528009696,
    "f4": -110.45409496678181
}

# Step 1: Create DataFrame from new input
new_X = pd.DataFrame([new_student_renamed])  # must be same order and columns as training
# Columns: ['score', 'age', 'x', 'y', 'z']

# Predict probabilities
probas = model.predict_proba(X_test_filtered)
probas_new_student = model.predict_proba(new_X)
print(f"len.probas:{len(probas)}")
print(f"len.probas_new_student:{len(probas_new_student)}")
print(f"probas_new_student:{probas_new_student}")

# Get top 3 predictions
top3_indices = np.argsort(probas, axis=1)[:, -3:][:, ::-1]
top3_school_ids = np.vectorize(index_to_id.get)(top3_indices)
#print(f"top3_school_ids:{top3_school_ids}")

# Show top 3 predictions for first 5 samples
#for i in range(len(top3_school_ids)):
#    print(f"Sample {i+1}: Top 3 predicted school IDs: {top3_school_ids[i]}")

# Encode ground truth labels
y_test_encoded = le.transform(y_test_filtered_raw)
# Map back to school UUIDs
top3_school_ids = np.vectorize(index_to_id.get)(top3_indices)

# Map from class index to UUID
index_to_uuid = dict(enumerate(le.classes_))

# For each sample in the test set
#for i in range(len(probas)):
#    # Get top 3 indices and probabilities
#    top3_indices = np.argsort(probas[i])[-3:][::-1]
#    top3_probs = probas[i][top3_indices]
#
#    print(f"\nSample {i + 1} - True label: {y_test_filtered_raw.iloc[i]}")
#    print("Top 3 Recommended Schools:")
#    for rank, (idx, prob) in enumerate(zip(top3_indices, top3_probs), start=1):
#        school_id = index_to_uuid[idx]
#        print(f"{rank}. {school_id} (Prob: {prob:.2f})")


# Step 1: Get top-1 predictions
print(f"len.probas:{len(probas)}")
top1_preds_indices = np.argmax(probas, axis=1)
print(f"top1_preds_indices:{top1_preds_indices}")
top1_preds_uuids = [index_to_uuid[i] for i in top1_preds_indices]

true_labels = y_test_filtered_raw.values.ravel()

print(len(true_labels))         # should be 2200
print(len(top1_preds_uuids))    # should also be 2200


# Step 2: Filter both to have the same set of UUIDs (optional, but safe)
unique_labels = sorted(list(set(true_labels) | set(top1_preds_uuids)))

# Step 3: Classification report
print("📊 Classification Report:")
print(classification_report(true_labels, top1_preds_uuids, labels=unique_labels, zero_division=0))

# Step 4: Confusion Matrix
cm = confusion_matrix(true_labels, top1_preds_uuids, labels=unique_labels)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, xticklabels=unique_labels, yticklabels=unique_labels, annot=False, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (Top-1 Prediction)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
#plt.show()


# For each sample in the test set
for i_std in range(len(probas_new_student)):
    # Get top 3 indices and probabilities
    top3_indices_new_student = np.argsort(probas_new_student[i_std])[-3:][::-1]
    top3_probs_new_student = probas_new_student[i_std][top3_indices_new_student]

    print(f"\nNew Student {i_std + 1} - True label: {y_test_filtered_raw.iloc[i_std]}")
    print("Top 3 Recommended Schools:")
    for rank_new_student, (idx_new_student, prob_new_student) in enumerate(zip(top3_indices_new_student, top3_probs_new_student), start=1):
        school_id = index_to_uuid[idx_new_student]
        print(f"{rank_new_student}. {school_id} idx_new_student:{idx_new_student} (Prob: {prob_new_student:.2f})")

# Convert to ONNX
initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]

options = {
    id(model): {
        'zipmap': False  # Disable seq(map(...)), get plain tensor(float)
    }
}
onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_type)
with open("school_recommender_xg_boost.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

#label encode
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_train_raw)

# Save it
with open("label_encoder_xg_boost.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
with open("label_encoder_xg_boost.pkl", "rb") as f:
    le = pickle.load(f)
with open("label_encoder_xg_boost.json", "w") as f:
    json.dump(le.classes_.tolist(), f)