# main.py
from db import Database
from service.student_service import StudentService
from service.school_service import SchoolService
import csv
import pandas as pd
import numpy as np


def latlng_to_xyz(lat, lng):
    lat_rad = np.radians(lat)
    lng_rad = np.radians(lng)
    x = np.cos(lat_rad) * np.cos(lng_rad)
    y = np.cos(lat_rad) * np.sin(lng_rad)
    z = np.sin(lat_rad)
    return x*1000, y*1000, z*1000

def convert_lat_lon():
    df = pd.read_csv('students.csv')

    # Loop through each row
    for idx, row in df.iterrows():
        x,y,z = latlng_to_xyz(row['student_lat'], row['student_lng'])
        #print(f"id:{row['junior_id']} ->> {x},{y},{z}")
        df.at[idx, 'x'] = x
        df.at[idx, 'y'] = y
        df.at[idx, 'z'] = z
        #df.at[idx, 'school_lat'] = float(school[0][2])  # school_lat
        #df.at[idx, 'school_lng'] = float(school[0][3])  # school_lng

    # Save updated CSV
    df.to_csv('students.csv', index=False)


def check_and_fix_missing_data():
    service = SchoolService()

    df = pd.read_csv('students.csv')

    # Show null summary before
    print("Null values before fixing:")
    print(df.isnull().sum())

    # Loop through each row
    for idx, row in df.iterrows():
        is_nan = False
        for col in df.columns:
            if pd.isna(row[col]):
                # Decide how to fill based on data type
                is_nan = True

                if df[col].dtype == 'object':  # text column
                    df.at[idx, col] = 'Unknown'
                else:  # numeric column
                    mean_value = df[col].mean()
                    df.at[idx, col] = mean_value
                #break
            #force type to float
            if col=='age' or col=='score':
                print(f"col:{col}")
                if type(row['age']) != float:
                    print(f"value:{row['age']} --> {float(row['age'])}")
                    df.at[idx, 'age'] = float(row['age'])
                if type(row['score']) != float:
                    df.at[idx, 'score'] = float(row['score'])

        if is_nan:
            #print(f"junior_id:{row['junior_id']}")
            pass
            '''
            school = service.get_latlng_school(row['junior_id'])
            print(f"db school: {school[0][2]}")
            df.at[idx, 'school_lat'] = float(school[0][2]) #school_lat
            df.at[idx, 'school_lng'] = float(school[0][3]) #school_lng 
            '''

    # Save updated CSV
    df.to_csv('students.csv', index=False)


def main():
    service = StudentService()
    students = service.list_students()

    #for student in students:
    #    print(student)

    # Write to CSV
    with open("students.csv", mode="w", newline="") as file:
        #fieldnames = ["id"]
        fieldnames = students[0].keys()
        print(f"fieldnames:{fieldnames}")
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(students)

    print("Data has been written to students.csv.")


if __name__ == "__main__":
    Database.init_pool(minconn=1, maxconn=5)
    main()
    check_and_fix_missing_data()
    convert_lat_lon()
    x, y, z = latlng_to_xyz(-6.273573798447577,106.93635761934014)
    print(f"x:{x}, y:{y}, z:{z}")
