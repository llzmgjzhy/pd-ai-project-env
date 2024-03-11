import time

import datetime
import numpy as np
from utils import *
from Mysql_connect import DatabaseConnection

with DatabaseConnection() as connection:
    cursor = connection.cursor()
    table_name = "us_waveform_sampledata_ai_analysis"
    if connection.is_connected():
        cursor.execute(
            f"DELETE FROM {table_name} WHERE id > 3"
        )
        connection.commit()
        print("Data delete successfully")
    else:
        print("Connection failed")