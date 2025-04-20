# main.py
from db import Database
from service.student_service import StudentService
import csv


def main():
    Database.init_pool(minconn=1, maxconn=5)

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
    main()