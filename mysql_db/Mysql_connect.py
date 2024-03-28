import mysql.connector
from mysql.connector import Error
import json
from pathlib import Path

current_dir = Path(__file__).resolve().parent

class DatabaseConnection:
    def __init__(self):
        # 打开文件并加载JSON数据
        with open(f"{current_dir}/DataBase.json", "r") as file:
            db_config = json.load(file)["database"]
        self.host_name = db_config["host_name"]
        self.db_name = db_config["db_name"]
        self.user_name = db_config["user_name"]
        self.user_password = db_config["user_password"]
        self.connection = None

    def __enter__(self):
        try:
            self.connection = mysql.connector.connect(
                host=self.host_name,
                database=self.db_name,
                user=self.user_name,
                password=self.user_password,
            )
            return self.connection
        except Error as e:
            print(f"Error connecting to MySQL: {e}")
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connection and self.connection.is_connected():
            self.connection.close()