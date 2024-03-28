import sys

sys.path.append("..")
from mysql_db.Mysql_connect import DatabaseConnection
import numpy as np
import pandas as pd
import datetime
import re
import datetime
from pathlib import Path

current_dir = Path(__file__).resolve().parent


def get_last_file():
    """
    acquire the last id from the last_file.txt file
     the file contains the last file name and station name for locating last operation
    """
    try:
        with open(f"{current_dir}/last_file.txt", "r") as file:
            content = file.read().strip().replace("\n", ",").split(",")
            filename = re.findall(r"\d+", content[0])[0]
            staname = re.findall(r"\d+", content[1])[0]
            return filename, staname
    except FileNotFoundError:
        return None


def get_last_file_wireless():
    """
    acquire the last file name,station name and id from the last_file_wireless.txt file
    return filename,staname,id
    """
    try:
        with open(f"{current_dir}/last_file_wireless.txt", "r") as file:
            content = file.read().strip().replace("\n", ",").split(",")
            filename = re.findall(r"\d+", content[0])[0]
            staname = re.findall(r"\d+", content[1])[0]
            id = re.findall(r"\d+", content[2])[0]
            return filename, staname, id
    except FileNotFoundError:
        return None


def save_last_file(filename, staname):
    """
    Save the last file name and station number to the last_id.txt file.
    Before saving the id, the old id will be removed.
    """
    station_number = re.findall(r"\d+", staname)[0]
    try:
        with open(f"{current_dir}/last_file.txt", "w") as file:
            file.write(f"file_name:{filename}\nstation_number:{station_number}")
            return "Last id saved to file."
    except IOError:
        print("Error: Unable to save last file info into file.")

def save_last_wireless_file(filename, staname,id):
    """
    Save the last file name and station number to the last_id.txt file.
    Before saving the id, the old id will be removed.
    """
    station_number = re.findall(r"\d+", staname)[0]
    try:
        with open(f"{current_dir}/last_file_wireless.txt", "w") as file:
            file.write(f"file_name:{filename}\nstation_number:{station_number}\nid:{id}")
            return "Last id saved to file."
    except IOError:
        print("Error: Unable to save last file info into file.")


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
            # cursor.execute(
            #     f"SELECT {', '.join(col_names)} FROM {table_name} WHERE CAST(SUBSTRING(FILE_NAME, LOCATE('AA_', FILE_NAME) + LENGTH('AA_'), LOCATE('.dat', FILE_NAME) - LOCATE('AA_', FILE_NAME) - LENGTH('AA_')) AS UNSIGNED) > {filename} OR CAST(SUBSTRING(FILE_NAME, LOCATE('AA_', FILE_NAME) + LENGTH('AA_'), LOCATE('.dat', FILE_NAME) - LOCATE('AA_', FILE_NAME) - LENGTH('AA_')) AS UNSIGNED) = {filename} AND CAST(SUBSTRING(STATION_NAME, LOCATE('测试',STATION_NAME)+CHAR_LENGTH('测试'),LOCATE('#',STATION_NAME)-LOCATE('测试',STATION_NAME)-CHAR_LENGTH('测试')) AS UNSIGNED) > {staname}"
            # )
            cursor.execute(
                f"SELECT {', '.join(col_names)} FROM {table_name} WHERE CAST(SUBSTRING(FILE_NAME, LOCATE('AA_', FILE_NAME) + LENGTH('AA_'), LOCATE('.dat', FILE_NAME) - LOCATE('AA_', FILE_NAME) - LENGTH('AA_')) AS UNSIGNED) > {filename} OR CAST(SUBSTRING(FILE_NAME, LOCATE('AA_', FILE_NAME) + LENGTH('AA_'), LOCATE('.dat', FILE_NAME) - LOCATE('AA_', FILE_NAME) - LENGTH('AA_')) AS UNSIGNED) = {filename} AND CAST(SUBSTRING(STATION_NAME, 1 ,LOCATE('开关柜',STATION_NAME)-1) AS UNSIGNED) > {staname}"
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


def get_filenames_jump_one(info_list, index):
    """
    acquire filenames list,if index is first,list conclude info_list[i+2] and info_list[i+4]; if last,conclude i-2 and i-4 ;whatever,filenames list will conclude 3 filename to match window models
    """
    if len(info_list) < 3:
        return []

    if index == len(info_list) - 1:
        filename_prev = info_list[index - 2]
        filename_next = info_list[index - 4]
    elif index == 0:
        filename_prev = info_list[index + 2]
        filename_next = info_list[index + 4]
    elif index == 1:
        filename_prev = info_list[index - 1]
        filename_next = info_list[index + 3]
    else:
        filename_prev = info_list[index - 2]
        filename_next = info_list[index + 2]
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

            table_name = "ai_alarm_results"
            model_name = "pd_" + map_type_code + "_" + pred_mode
            if predicted.item() == 1:
                cursor.execute(
                    f"INSERT INTO {table_name} (file_name, device_name,station_name, model_name, model_version, alarm_result,pd_type,operation_time) VALUES ('{filename}', '开关柜', '{staname}','{model_name}', '0.1', '{prob.item()}','{predicted.item()}','{operation_time}')"
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


def save_pred_result_voltage(
    prob, predicted, filename, pos_name, pos_code, map_type_code, pred_mode,id = 0
):
    """
    Save the prediction result to the database.
    """

    # db connect
    table_name = "us_feature_ai_analysis"
    with DatabaseConnection() as connection:
        cursor = connection.cursor()
        if connection.is_connected():
            # Save the prediction result to the database.
            operation_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]

            cursor.execute(
                f"SELECT * FROM {table_name} WHERE file_name = '{filename}' AND measure_position_name = '{pos_name}'"
            )
            existing_data = cursor.fetchone()

            if existing_data:
                # Update existing data
                cursor.execute(
                    f"UPDATE {table_name} SET pd_type_{pred_mode} = {predicted.item()}, pd_prob_{pred_mode} = {prob.item()} WHERE file_name = '{filename}' AND measure_position_name = '{pos_name}'"
                )
            else:
                # Insert new data
                cursor.execute(
                    f"INSERT INTO {table_name} (file_name, measure_position_name,measure_position_code, pd_type_{pred_mode}, pd_prob_{pred_mode}, operation_time) VALUES ('{filename}', '{pos_name}', '{pos_code}',{predicted.item()}, {prob.item()}, '{operation_time}')"
                )

            # Insert alarm data
            table_name = "ai_alarm_results"
            model_name = "pd_" + map_type_code + "_" + pred_mode

            if predicted.item() == 1:
                cursor.execute(
                    f"INSERT INTO {table_name} (file_name, device_name,station_name, model_name, model_version, alarm_result,pd_type,operation_time) VALUES ('{filename}', '开关柜', '{pos_name}','{model_name}', '0.1', '{prob.item()}','{predicted.item()}','{operation_time}')"
                )
            connection.commit()
        cursor.close()

    # Save the latest file name and station name to the last_id.txt file.
    if map_type_code == "0x31" and pred_mode == "w":
        save_last_wireless_file(filename, pos_name,id)
    print(
        "文件",
        filename,
        "+",
        pos_name,
        "+",
        map_type_code,
        "+",
        pred_mode,
        "预测成功，预测结果已保存到数据库。",
    )


def ai_evaluation_database(sensor_version, ai_model_version, data_size, model_accuracy):
    table_name = "ai_model_evaluation"
    with DatabaseConnection() as connection:
        cursor = connection.cursor()
        if connection.is_connected():
            # Save the prediction result to the database.
            date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]

            cursor.execute(
                f"INSERT INTO {table_name} (datetime, Sensor_version,Ai_model_version, data_size, model_accuracy) VALUES ('{date_time}', 'AE', '0.1','{data_size}', '0.7')"
            )
            connection.commit()
