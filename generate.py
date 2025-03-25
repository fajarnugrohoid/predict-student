import pandas as pd
import numpy as np
import random
import csv

# Generate 100 rows of school data
num_schools = 100
school_data = []

for i in range(num_schools):
    school_data.append([
        random.randint(50, 100),  # math_score
        random.randint(50, 100),  # english_score
        random.randint(50, 100),  # bahasa_score
        round(random.uniform(-7.0, -6.9), 6),  # latitude
        round(random.uniform(107.7, 107.8), 6),  # longitude
        round(random.uniform(-7.0, -6.9), 6),  # school_latitude
        round(random.uniform(107.7, 107.8), 6),  # school_longitude
        round(random.uniform(100, 1000), 2),  # distance
        f"School_{i}"  # accepted_school
    ])

# Save to CSV
csv_filename = "school_data.csv"
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["math_score", "english_score", "bahasa_score", "latitude", "longitude",
                     "school_latitude", "school_longitude", "distance", "accepted_school"])
    writer.writerows(school_data)

print(f"CSV file '{csv_filename}' created successfully!")