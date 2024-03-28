from dataloader import Base_DataLoader_Tst,Base_DataLoader_Trn
from mysql_db import DatabaseConnection
import numpy as np
import pandas as pd


class Dataloader_Tst(Base_DataLoader_Tst):
    def __init__(self, info_table_name, data_table_name, filename, staname):
        super(Dataloader_Tst, self).__init__(info_table_name, data_table_name)

        with DatabaseConnection() as db:
            self.conn = db
            self.cur = self.conn.cursor()
            self.tst, self.pulse_count = self.read_data(filename, staname)

    def read_info(self):
        pass

    def read_data(self, filename, staname):
        col_names = ["id", "FILE_NAME", "STATION_NAME"]
        col_names.extend(["col" + str(i) for i in range(1, 61)])
        if self.conn.is_connected():
            cursor = self.conn.cursor()
            file_name = filename
            sta_name = staname

            # 构建查询语句
            query = f"SELECT {', '.join(col_names)} FROM {table_name} WHERE FILE_NAME = '{file_name}' AND STATION_NAME = '{sta_name}'"  # \ AND SUBSTR(STATION_NAME, 1, 4) = '{sta_name}'"
            # 执行查询
            cursor.execute(query)

            # 获取查询结果
            newdata = cursor.fetchall()
            # 将datas转成ndarray
            newdata = np.array(newdata)
            newdata = newdata[:, 3:].astype(np.float32)
            # 计算newdata总值
            pulse_count = np.sum(np.sum(newdata, axis=0))
            newdata = np.expand_dims(newdata, axis=0)  # 脉冲数量
            # 关闭cursor
            cursor.close()
        return newdata, pulse_count
    
class test_cls():
    def __init__(self) -> None:
        pass

    def hello(self):
        print("Hello World")

class Dataloader_Trn(Base_DataLoader_Trn):
    def __init__(self, info_table_name, data_table_name, filename, staname):
        super(Dataloader_Tst, self).__init__(info_table_name, data_table_name)

        with DatabaseConnection() as db:
            self.conn = db
            self.cur = self.conn.cursor()
            self.tst, self.pulse_count = self.read_data(filename, staname)

    def read_info(self):
        pass

    def read_data(self, filename, staname):
        col_names = ["id", "FILE_NAME", "STATION_NAME"]
        col_names.extend(["col" + str(i) for i in range(1, 61)])
        if self.conn.is_connected():
            cursor = self.conn.cursor()
            file_name = filename
            sta_name = staname

            # 构建查询语句
            query = f"SELECT {', '.join(col_names)} FROM {table_name} WHERE FILE_NAME = '{file_name}' AND STATION_NAME = '{sta_name}'"  # \ AND SUBSTR(STATION_NAME, 1, 4) = '{sta_name}'"
            # 执行查询
            cursor.execute(query)

            # 获取查询结果
            newdata = cursor.fetchall()
            # 将datas转成ndarray
            newdata = np.array(newdata)
            newdata = newdata[:, 3:].astype(np.float32)
            # 计算newdata总值
            pulse_count = np.sum(np.sum(newdata, axis=0))
            newdata = np.expand_dims(newdata, axis=0)  # 脉冲数量
            # 关闭cursor
            cursor.close()
        return newdata, pulse_count