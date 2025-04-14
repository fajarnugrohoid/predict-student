import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from math import radians, sin, cos, sqrt, atan2
from haversine import haversine, Unit
from sklearn.neighbors import BallTree
import matplotlib.pyplot as plt
import json



'''
# Function to calculate Haversine distance
def haversine(lat1, lon1, lat2, lon2):
    R = 6373  # Radius of the Earth in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c
'''
# read from csv file
df = pd.read_csv('student_data.csv')
df_school = pd.read_csv('school_data.csv')

# Convert latitude and longitude to radians for BallTree
df_school["latitude_rad"] = np.deg2rad(df_school["school_latitude"])
df_school["longitude_rad"] = np.deg2rad(df_school["school_longitude"])

# Compute distance using each student's accepted school's coordinates
#df['distance'] = df.apply(lambda row: haversine( (row['latitude'], row['longitude']), (row['school_latitude'], row['school_longitude']), unit=Unit.METERS), axis=1)
# Convert latitude/longitude to radians for BallTree
school_locations = np.column_stack((df_school["latitude_rad"], df_school["longitude_rad"]))

tree = BallTree(school_locations, metric="haversine")

# Student's location (Convert to radians)
student_location = np.deg2rad([[-6.932284674339599,107.78140745962706]])

# Find the nearest school
distances, indices = tree.query(student_location, k=3)

# Extract the nearest school details
nearest_index0 = indices[0][0]
nearest_index1 = indices[0][1]
nearest_index2 = indices[0][2]
nearest_distance = distances[0][0] * 6371000  # Convert radians to meters
nearest_school0 = df_school.iloc[nearest_index0]["accepted_school"]
nearest_school1 = df_school.iloc[nearest_index1]["accepted_school"]
nearest_school2 = df_school.iloc[nearest_index2]["accepted_school"]

feature_columns = ['math_score', 'english_score', 'bahasa_score', 'distance']

print(f"indices:{indices}")
print(f"nearest_index:{nearest_index0}")
print(f"nearest_index:{nearest_index1}")
print(f"nearest_distance:{nearest_distance}")
print(f"nearest_school0:{nearest_school0}")
print(f"nearest_school1:{nearest_school1}")
print(f"nearest_school2:{nearest_school2}")

# Encode target variable
label_encoder = LabelEncoder()
df['accepted_school'] = label_encoder.fit_transform(df['accepted_school'])

# Split features and target
X = df[['math_score', 'english_score', 'bahasa_score',
                    'distance']]
y = df['accepted_school']

feature_names = X.columns.tolist()

class_names = list(map(str, y.unique()))
print(f"class_names:{class_names}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Decision Tree Model
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
clf.fit(X_train, y_train)



plt.figure(figsize=(50, 40))  # Set figure size
plot_tree(
    clf,
    filled=True,  # Fill nodes with colors
    feature_names=feature_names,  # Label features
    class_names=class_names  # Label class names
)
plt.show()



# Predictions
y_pred = clf.predict(X_test)

# Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Example Prediction
#nanti hrus di loop 3-5 sekolah terdekat,
distance_new_student = haversine( (-6.932284674339599,107.78140745962706), (-6.928319,107.779412), unit=Unit.METERS)
print(f"distance_new_student:{distance_new_student}")
new_student = np.array([[97,91,58, distance_new_student]])  # Example input

# Predict probabilities for each school
probabilities = clf.predict_proba(new_student)
print(f"probabilities:{probabilities}")

# Get class names (school names)
school_names = clf.classes_
print(f"school_names:{school_names}")

# Get top 3 schools with highest probability
top_3_indices = np.argsort(probabilities[0])[-3:][::-1]  # Sort and get top 3
top_3_schools = [(school_names[i], probabilities[0][i]) for i in top_3_indices]

# Print results
print("Top 3 Recommended Schools:")
for rank, (school, prob) in enumerate(top_3_schools, 1):
    print(f"{rank}. {school} (Probability: {prob:.2f})")

new_student_scaled = scaler.transform(new_student)
model = clf.predict(new_student_scaled)
predicted_school = label_encoder.inverse_transform(model)
print(f'Model: {model}')
print(f'Predicted School: {predicted_school[0]}')