import logging
from config import DATABASE_URL

logger = logging.getLogger("ai-engine.database")

def get_db_connection():
    """
    Tạo kết nối tới PostgreSQL và đăng ký pgvector adapter.
    Hỗ trợ linh hoạt cả psycopg3 và psycopg2-binary.
    """
    try:
        try:
            import psycopg
            from pgvector.psycopg import register_vector
            conn = psycopg.connect(DATABASE_URL)
            register_vector(conn)
            return conn
        except ImportError:
            import psycopg2
            from pgvector.psycopg2 import register_vector
            conn = psycopg2.connect(DATABASE_URL)
            register_vector(conn)
            return conn
    except Exception as e:
        logger.warning(f"Chưa thể kết nối CSDL tại {DATABASE_URL}: {e}")
        return None
