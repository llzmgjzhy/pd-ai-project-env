import time

import datetime
import numpy as np
# from utils import *
import sys
sys.path.append("..")
from Mysql_connect import DatabaseConnection
import pandas as pd
import os
from pathlib import Path


with DatabaseConnection() as connection:
    cursor = connection.cursor()
    table_name = "ai_alarm_results"
    if connection.is_connected():
        cursor.execute(
            f"DELETE FROM {table_name} WHERE id > 0"
        )
        connection.commit()
        print("Data delete successfully")
    else:
        print("Connection failed")

# pwd = os.getcwd()
# dir_path = os.path.abspath(os.path.dirname(pwd))

# curr_path = os.path.join(dir_path, "monitor_analysis")
# print(curr_path)