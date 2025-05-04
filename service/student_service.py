# service/student_service.py
from repository.student_repository import StudentRepository

class StudentService:
    def __init__(self):
        self.repo = StudentRepository()

    def list_students(self):
        result = []
        students = self.repo.get_all_students()
        for student in students:
            if student["selection_current"]==1:
                distance =student["distance_1"]
            elif student["selection_current"]==2:
                distance = student["distance_2"]
            else:
                distance = student["distance_3"]

            temp = {
                    "id": student["id"], "junior_id": student["junior_id"],
                    "score": student["score"],"age": student["age"],
                    "student_lat": student["student_lat"], "student_lng": student["student_lng"],
                    "accepted_school_id" : student["accepted_school_id"],
                    "school_lat": student["school_lat"], "school_lng": student["school_lng"],
            #        "distance": distance,
                    "status": student["status"]
                    }
            result.append(temp)
        return result