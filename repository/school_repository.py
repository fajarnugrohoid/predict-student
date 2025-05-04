# repository/school_repository.py
from db import Database

class SchoolRepository:
    def get_school_by_id(self, id):
        db = Database()
        try:
            query = ("SELECT id, name, coordinate_lat,coordinate_lng FROM registration.schools WHERE id= %s")
            return db.fetch_all(query, (str(id),))
        finally:
            db.close()
