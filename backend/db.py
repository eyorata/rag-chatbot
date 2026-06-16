import psycopg2
from psycopg2.pool import SimpleConnectionPool
from config import settings

pool = SimpleConnectionPool(1, 10, settings.database_url)

def get_conn():
    return pool.getconn()

def put_conn(conn):
    pool.putconn(conn)