import pandas as pd
import numpy as np
import random
import csv

# Load the CSV into Pandas DataFrame
df = pd.read_csv("student_data.csv")


for index, row in df.iterrows():
    if row["accepted_school"] == "VHS1":
        df.at[index, "school_latitude"] = -6.92396133741657
        df.at[index, "school_longitude"] = 107.78009223685055

    if row["accepted_school"] == "VHS2":
        df.at[index, "school_latitude"] = -6.915544889764089
        df.at[index, "school_longitude"] = 107.71838571509048

    if row["accepted_school"] == "VHS3":
        df.at[index, "school_latitude"] = -6.951993691980129
        df.at[index, "school_longitude"] = 107.7275585867526

    if row["accepted_school"] == "VHS4":
        df.at[index, "school_latitude"] = -6.9529526870044025
        df.at[index, "school_longitude"] = 107.71367777975831

    if row["accepted_school"] == "VHS5":
        df.at[index, "school_latitude"] = -6.956058741464109
        df.at[index, "school_longitude"] = 107.75054745698117



df.to_csv("student_data.csv", index=False)
