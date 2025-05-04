# repository/student_repository.py
from db import Database

class StudentRepository:
    def get_all_students(self):
        db = Database()
        try:
            query = ("SELECT rr.id,rr.junior_id, rr.score, rr.age, rr.selection_current, "
                     "rr.distance_1, rr.distance_2, rr.distance_3, "
                     "rj.coordinate_lat AS student_lat,rj.coordinate_lng AS student_lng,rr.accepted_choice_id, "
                     "rr.accepted_school_id,rs.coordinate_lat AS school_lat,rs.coordinate_lng AS school_lng, rr.status "
                     "FROM registration.registration rr "
                     "INNER JOIN ref.junior_data rj ON (rj.id=rr.junior_id)"
                     "LEFT JOIN registration.schools rs ON (rs.id=rr.accepted_school_id)"
                     "WHERE rr.status='accepted' AND score IS NOT NULL"
            #         "LIMIT 10"
            )
            return db.fetch_all(query)
        finally:
            db.close()

    def insert_student(self, id, name):
        db = Database()
        try:
            query = "INSERT INTO students (id, name) VALUES (%s, %s)"
            db.execute(query, (id, name))
        finally:
            db.close()

    def update_student(self, id, new_name):
        db = Database()
        try:
            query = "UPDATE students SET name = %s WHERE id = %s"
            db.execute(query, (new_name, id))
        finally:
            db.close()

    def delete_student(self, id):
        db = Database()
        try:
            query = "DELETE FROM students WHERE id = %s"
            db.execute(query, (id,))
        finally:
            db.close()
