import sys

sys.path.append("..")
from Mysql_connect import DatabaseConnection
import numpy as np
import pandas as pd
import datetime


def get_last_id():
    """
    acquire the last id from the last_id.txt file
     so the file only contains the last id
    """
    try:
        with open("last_id.txt", "r") as file:
            last_id = int(file.read().strip())
            return last_id
    except FileNotFoundError:
        return None


def save_last_id(last_id):
    """
    Save the last id to the last_id.txt file.
    Before saving the id, the old id will be removed.
    """
    try:
        with open("last_id.txt", "w") as file:
            file.write(str(last_id))
            return "Last id saved to file."
    except IOError:
        print("Error: Unable to save last id to file.")


def read_us_prpd_info():
    """
    Read the data info from the corresponding database.
    """

    table_name = "us_waveform_prpd_info"
    col_names = ["id", "FILE_NAME", "STATION_NAME"]
    last_id = get_last_id()
    # db connect
    with DatabaseConnection() as connection:
        cursor = connection.cursor()
        if connection.is_connected():
            cursor.execute(
                f"SELECT {', '.join(col_names)} FROM {table_name} WHERE id > {last_id}"
            )
            data = cursor.fetchall()[:10]
            if data:
                latest_id = max(data, key=lambda x: x[0])[0]
                data = pd.DataFrame(data, columns=col_names)
            else:
                print("No data found.")
                return None
        cursor.close()
    return data, latest_id


def search_new_data():
    # db connect
    with DatabaseConnection() as connection:
        cursor = connection.cursor()
        if connection.is_connected():
            last_id = get_last_id()
            if last_id is not None:
                cursor.execute("SELECT * FROM data_table")
            else:
                print("The last_id is unreasonable. Please check the last_id.txt file.")
            cursor.execute("SELECT * FROM data_table")
            data = cursor.fetchall()
        cursor.close()

    return data


def save_pred_result(prob, predicted, filename, staname, map_type_code, latest_id):
    """
    Save the prediction result to the database.
    """
    # db connect
    table_name = "us_waveform_sampledata_ai_analysis"
    with DatabaseConnection() as connection:
        cursor = connection.cursor()
        if connection.is_connected():
            # Save the prediction result to the database.
            operation_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
            cursor.execute(
                f"INSERT INTO {table_name} (file_name, station_name, map_type, pd_type, pd_prob, operation_time) VALUES ('{filename}', '{staname}', '{map_type_code}', {predicted.item()}, {prob.item()}, '{operation_time}')"
            )
            connection.commit()
        cursor.close()
    # Save the latest id to the last_id.txt file.
    save_last_id(latest_id)
    print("文件", filename, "+", staname, "预测成功，预测结果已保存到数据库。")
