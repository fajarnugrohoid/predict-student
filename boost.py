import pandas as pd
import numpy as np
import xg_boost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv('student_data.csv')

# Define features and target
feature_columns = ["math_score", "english_score", "bahasa_score", "distance"]
X = df[feature_columns]  # Features
y = df["accepted_school"]  # Target

# Encode target labels as numbers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Converts school names into numerical labels

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Define XGBoost model for multi-class classification
xgb_model = xgb.XGBClassifier(
    objective='multi:softprob',  # Multi-class classification
    num_class=len(label_encoder.classes_),  # Number of unique schools
    eval_metric='mlogloss',
    use_label_encoder=False,
    max_depth=5,
    learning_rate=0.1,
    n_estimators=100
)
#df["distance"] *= 50  # Increase the weight of distance in the model


sample_weights = df["distance"] / df["distance"].max()

# Train the model
xgb_model.fit(X_train, y_train, sample_weight=sample_weights[:len(y_train)])
#xgb_model.fit(X_train, y_train)

# Predict on test set
y_pred = xgb_model.predict(X_test)

# Convert predicted labels back to school names
y_pred_labels = label_encoder.inverse_transform(y_pred)

# Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Get feature importance scores
importance = xgb_model.feature_importances_
print(f"\nimportance:{importance}")


# Plot feature importance
plt.barh(feature_columns, importance)
plt.xlabel("Feature Importance")
plt.ylabel("Feature Name")
plt.title("XGBoost Feature Importance")
plt.show()

# Function to predict top 3 recommended schools
def recommend_schools(student_features):
    print(f"student_features:{student_features}")
    probabilities = xgb_model.predict_proba([student_features])[0]
    top_3_indices = np.argsort(probabilities)[-3:][::-1]  # Get top 3 indices
    top_3_schools = [(label_encoder.inverse_transform([i])[0], probabilities[i]) for i in top_3_indices]
    return top_3_schools

# Example: Predict for a new student
new_student = [97,91,58, 350]
top_schools = recommend_schools(new_student)

print("\nTop 3 Recommended Schools:")
for rank, (school, prob) in enumerate(top_schools, 1):
    print(f"{rank}. {school} (Probability: {prob:.2f})")
