# db.py
from psycopg2.pool import ThreadedConnectionPool
import psycopg2.extras
from config import DB_CONFIG

class Database:
    _pool = None

    @classmethod
    def init_pool(cls, minconn=1, maxconn=5):
        if cls._pool is None:
            cls._pool = ThreadedConnectionPool(minconn, maxconn, **DB_CONFIG)

    def __init__(self):
        if Database._pool is None:
            raise Exception("Connection pool is not initialized. Call init_pool() first.")
        self.conn = Database._pool.getconn()
        self.cursor = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    def fetch_all(self, query, params=None):
        self.cursor.execute(query, params or ())
        return self.cursor.fetchall()

    def execute(self, query, params=None):
        self.cursor.execute(query, params or ())
        self.conn.commit()

    def close(self):
        self.cursor.close()
        Database._pool.putconn(self.conn)
