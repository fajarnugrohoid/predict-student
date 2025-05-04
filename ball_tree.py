import numpy as np
from sklearn.neighbors import BallTree
import pandas as pd

# Earth's radius in km
EARTH_RADIUS = 6371

# Load data
df = pd.read_csv("students.csv")
features = ['school_lat', 'school_lng']
schools_df = df[features]


# Convert lat/lon to radians for haversine
school_locations_df_rad = np.radians(schools_df.to_numpy())

# Build BallTree with Haversine metric
#tree = BallTree(school_locations_rad, metric='haversine')
tree = BallTree(school_locations_df_rad, metric='haversine')
# Student's info
student_lat = -6.985773
student_lon = 107.839561
student_age = 8

# Query 5 nearest schools
student_location_rad = np.radians([[student_lat, student_lon]])

# distances in radians, need to multiply by Earth's radius to get km
distances_rad, indices = tree.query(student_location_rad, k=6)
distances_km = distances_rad[0] * EARTH_RADIUS

# Get nearby schools' indices
nearby_school_indices = indices[0]

print("Nearby schools (before filtering):")
for idx, dist in zip(nearby_school_indices, distances_km):
    school_id = df.iloc[idx]['accepted_school_id']
    #print(f"School {idx}: Distance = {dist:.2f} km, Age range = {schools[idx,2]}-{schools_[idx,3]}")
    print(f" Idx {idx}, School {school_id}: Distance = {dist:.2f} km")
