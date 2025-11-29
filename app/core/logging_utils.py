import logging
from logging.handlers import RotatingFileHandler
import time
import mysql.connector
from app.config import config

class DatabaseLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.conn = None
        self.cursor = None
        self.reconnect()

    def reconnect(self):
        try:
            if self.conn:
                try: self.conn.close()
                except: pass
            self.conn = mysql.connector.connect(
                host=config.get("MYSQL_HOST"),
                port=config.get("MYSQL_PORT"),
                database=config.get("MYSQL_DB"),
                user=config.get("MYSQL_USER"),
                password=config.get("MYSQL_PASSWORD"),
            )
            self.cursor = self.conn.cursor()
        except Exception as e:
            print(f"数据库连接失败: {e}")
            self.conn, self.cursor = None, None

    def emit(self, record: logging.LogRecord):
        if self.conn is None or not self.conn.is_connected():
            self.reconnect()
            if self.conn is None:
                return
        try:
            message = self.format(record)
            # TEXT/mediumtext 安全截断
            max_length = 65000
            if len(message) > max_length:
                message = message[:max_length-3] + "..."
            ts = int(time.time()*1000)
            sql = """
            INSERT INTO logging_event (timestmp, formatted_message, logger_name, level_string, thread_name, application_name)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            params = (ts, message, record.name, record.levelname, record.threadName, 'transcribe-service')
            self.cursor.execute(sql, params)
            self.conn.commit()
        except Exception as e:
            print(f"写入日志到数据库失败: {e}")
            try: self.reconnect()
            except: pass

def setup_logging():
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)

    fh = RotatingFileHandler("transcribe_service.log", maxBytes=10*1024*1024, backupCount=5)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    try:
        dbh = DatabaseLogHandler()
        dbh.setFormatter(fmt)
        dbh.setLevel(logging.INFO)
        root.addHandler(dbh)
        logging.info("数据库日志处理器已添加")
    except Exception as e:
        logging.error(f"添加数据库日志处理器失败: {e}")
