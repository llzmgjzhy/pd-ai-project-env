"""
MySQL建表操作
"""
import os
import pymysql
from pymysql import Error
import struct
import json


# 连接数据库
def create_connection(host_name, user_name, user_password, db_name):
    connection = None
    try:
        connection = pymysql.connect(
            host=host_name, user=user_name, password=user_password, database=db_name
        )
        print("Connection to MySQL DB successful")
    except Error as e:
        print(f"The error '{e}' occurred")

    return connection


# 创建数据表
def create_table(connection, create_table_sql, table_name):
    cursor = connection.cursor()  # 创建游标对象
    try:
        cursor.execute(create_table_sql)
        print("Table {} created successfully".format(table_name))
    except Error as e:
        print(f"The error '{e}' occurred")

def create_Ai_model_evaluation(connection):
    # 从JSON文件读取表结构
    with open("Ai_Table.json", "r") as file:
        table_structure = json.load(file)
    # 构建SQL建表语句
    create_table_sql = "CREATE TABLE IF NOT EXISTS Ai_model_evaluation ("  # 创建数据信息表
    create_table_sql += "id INT AUTO_INCREMENT PRIMARY KEY, "  # 添加自增主键列
    for column, attrs in table_structure["Ai_model_evaluation"].items():
        column_type = attrs["type"]
        if column_type == "VARCHAR":
            column_type += f"({attrs['b_length']})"
        create_table_sql += f"{column} {column_type}, "
    # 去掉最后一个逗号，并添加闭括号
    create_table_sql = create_table_sql.rstrip(", ") + ");"

    create_table(connection, create_table_sql, "Ai_model_evaluation")

def create_Ai_alarm_results(connection):
    # 从JSON文件读取表结构
    with open("Ai_Table.json", "r") as file:
        table_structure = json.load(file)
    # 构建SQL建表语句
    create_table_sql = "CREATE TABLE IF NOT EXISTS Ai_alarm_results ("  # 创建数据信息表
    create_table_sql += "id INT AUTO_INCREMENT PRIMARY KEY, "  # 添加自增主键列
    for column, attrs in table_structure["Ai_alarm_results"].items():
        column_type = attrs["type"]
        if column_type == "VARCHAR":
            column_type += f"({attrs['b_length']})"
        create_table_sql += f"{column} {column_type}, "
    # 去掉最后一个逗号，并添加闭括号
    create_table_sql = create_table_sql.rstrip(", ") + ");"

    create_table(connection, create_table_sql, "Ai_alarm_results")



def main():
    # 打开文件并加载JSON数据
    with open("DATABASE.json", "r") as file:
        db_config = json.load(file)["database"]
    # Database credentials
    host_name = db_config["host_name"]
    user_name = db_config["user_name"]
    user_password = db_config["user_password"]
    db_name = db_config["db_name"]
    # 连接数据库
    connection = create_connection(host_name, user_name, user_password, db_name)
    # 创建数据表
    if connection:
        create_Ai_model_evaluation(connection)
        create_Ai_alarm_results(connection)


    if connection:
        connection.close()


if __name__ == "__main__":
    main()
