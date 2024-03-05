import sys

sys.path.append("..")
from Mysql_connect import DatabaseConnection
import numpy as np
import pandas as pd
import datetime
import re
import datetime


def get_last_file():
    """
    acquire the last id from the last_file.txt file
     the file contains the last file name and station name for locating last operation
    """
    try:
        with open("last_file.txt", "r") as file:
            content = file.read().strip().replace("\n", ",").split(",")
            filename = re.findall(r"\d+", content[0])[0]
            staname = re.findall(r"\d+", content[1])[0]
            return filename, staname
    except FileNotFoundError:
        return None


def save_last_file(filename, staname):
    """
    Save the last file name and station number to the last_id.txt file.
    Before saving the id, the old id will be removed.
    """
    station_number = re.findall(r"\d+", staname)[0]
    try:
        with open("last_file.txt", "w") as file:
            file.write(f"file_name:{filename}\nstation_number:{station_number}")
            return "Last id saved to file."
    except IOError:
        print("Error: Unable to save last id to file.")


def read_us_data_info(table_name="us_waveform_prpd_info_bak"):
    """
    Read the data info from the corresponding database.
    """

    col_names = ["id", "FILE_NAME", "STATION_NAME"]
    filename, staname = get_last_file()
    # db connect
    with DatabaseConnection() as connection:
        cursor = connection.cursor()
        if connection.is_connected():
            cursor.execute(
                f"SELECT {', '.join(col_names)} FROM {table_name} WHERE CAST(SUBSTRING(FILE_NAME, LOCATE('AA_', FILE_NAME) + LENGTH('AA_'), LOCATE('.dat', FILE_NAME) - LOCATE('AA_', FILE_NAME) - LENGTH('AA_')) AS UNSIGNED) > {filename} OR CAST(SUBSTRING(FILE_NAME, LOCATE('AA_', FILE_NAME) + LENGTH('AA_'), LOCATE('.dat', FILE_NAME) - LOCATE('AA_', FILE_NAME) - LENGTH('AA_')) AS UNSIGNED) = {filename} AND CAST(SUBSTRING(STATION_NAME, LOCATE('测试',STATION_NAME)+CHAR_LENGTH('测试'),LOCATE('#',STATION_NAME)-LOCATE('测试',STATION_NAME)-CHAR_LENGTH('测试')) AS UNSIGNED) > {staname}"
            )
            data = cursor.fetchall()
            if data:
                data = pd.DataFrame(data, columns=col_names)
            else:
                print("No data found.")
                return []  # Return an empty list instead of None when no data is found
        cursor.close()
    return data


def get_filenames(info_list, index):
    """
    acquire filenames list,if index is first,list conclude info_list[i+1] and info_list[i+2]; if last,conclude i-1 and i-2 ;whatever,filenames list will conclude 3 filename to match window models
    """
    if len(info_list) < 3:
        return []

    if index == len(info_list) - 1:
        filename_prev = info_list[index - 1]
        filename_next = info_list[index - 2]
    elif index == 0:
        filename_prev = info_list[index + 1]
        filename_next = info_list[index + 2]
    else:
        filename_prev = info_list[index - 1]
        filename_next = info_list[index + 1]
    filenames = [filename_prev, info_list[index], filename_next]

    return filenames


def save_pred_result(prob, predicted, filename, staname, map_type_code, pred_mode):
    """
    Save the prediction result to the database.
    """
    if map_type_code == "0x35":
        map_type = "prpd"
    elif map_type_code == "0x36":
        map_type = "prps"

    # db connect
    table_name = "us_waveform_sampledata_ai_analysis"
    with DatabaseConnection() as connection:
        cursor = connection.cursor()
        if connection.is_connected():
            # Save the prediction result to the database.
            operation_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]

            cursor.execute(
                f"SELECT * FROM {table_name} WHERE file_name = '{filename}' AND station_name = '{staname}'"
            )
            existing_data = cursor.fetchone()

            if existing_data:
                # Update existing data
                cursor.execute(
                    f"UPDATE {table_name} SET pd_type_{map_type}_{pred_mode} = {predicted.item()}, pd_prob_{map_type}_{pred_mode} = {prob.item()} WHERE file_name = '{filename}' AND station_name = '{staname}'"
                )
            else:
                # Insert new data
                cursor.execute(
                    f"INSERT INTO {table_name} (file_name, station_name, pd_type_{map_type}_{pred_mode}, pd_prob_{map_type}_{pred_mode}, operation_time) VALUES ('{filename}', '{staname}', {predicted.item()}, {prob.item()}, '{operation_time}')"
                )
            connection.commit()
        cursor.close()

    # Save the latest file name and station name to the last_id.txt file.
    if map_type_code == "0x36" and pred_mode == "w":
        save_last_file(filename, staname)
    print(
        "文件",
        filename,
        "+",
        staname,
        "+",
        map_type_code,
        "+",
        pred_mode,
        "预测成功，预测结果已保存到数据库。",
    )
