# service/student_service.py
from repository.school_repository import SchoolRepository

class SchoolService:
    def __init__(self):
        self.repo = SchoolRepository()

    def get_latlng_school(self, id):
        school = self.repo.get_school_by_id(id)

        return school