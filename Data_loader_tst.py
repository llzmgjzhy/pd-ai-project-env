from Mysql_connect import DatabaseConnection
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class Dataset_load_tst(Dataset):
    def __init__(self, map_type_code, filename, staname):
        # 连接数据库
        with DatabaseConnection() as db:
            self.conn = db
            self.cur = self.conn.cursor()
            if map_type_code == "0x35":
                self.tst, self.pulse_count = self.read_us_prpd_sampledata(filename, staname)
            elif map_type_code == "0x36":
                self.tst, self.pulse_count = self.read_us_prps_sampledata(filename, staname)

    def read_us_prpd_info(self):
        # 表名
        table_name = "us_waveform_prpd_info"
        # 列名
        col_names = ["id", "FILE_NAME", "STATION_NAME"]
        # 提取数据中col_names的数据
        self.cur.execute(f"SELECT {', '.join(col_names)} FROM {table_name}")

        rows = self.cur.fetchall()
        # 转成DataFrame
        df = pd.DataFrame(rows, columns=col_names)
        return df

    def read_us_prpd_sampledata(self, filename, staname):
        # info_list = self.read_us_prpd_info()[5000:5020]
        # filename = info_list["FILE_NAME"][5000]
        # staname = info_list["STATION_NAME"][5000]
        # 表名
        table_name = "us_waveform_prpd_sampledata"
        # 列名
        col_names = ["id", "FILE_NAME", "STATION_NAME"]
        # 列名还有coll_1到coll_60
        col_names.extend(["col_" + str(i) for i in range(1, 61)])
        rebuild_data = []
        if self.conn.is_connected():
            cursor = self.conn.cursor()
            file_name = filename
            sta_name = staname

            # 构建查询语句
            query = f"SELECT {', '.join(col_names)} FROM {table_name} WHERE FILE_NAME = '{file_name}'"# AND SUBSTR(STATION_NAME, 1, 4) = '{sta_name}'"
            # 执行查询
            cursor.execute(query)

            # 获取查询结果
            newdata = cursor.fetchall()
            # 将datas转成ndarray
            newdata = np.array(newdata)
            newdata = newdata[:80, 3:].astype(np.float32)
            # 计算newdata总值
            pulse_count = np.sum(np.sum(newdata, axis=0))
            newdata = np.expand_dims(newdata, axis=0)  # 脉冲数量
            # 关闭cursor
            cursor.close()
        return newdata, pulse_count
    
    
    def read_us_prpd_sampledata_custom(self, list):
        info_list = list
        filename = info_list["FILE_NAME"]
        staname = info_list["STATION_NAME"]

        table_name = "us_waveform_prpd_sampledata"
        col_names = ["id", "FILE_NAME", "STATION_NAME"]
        col_names.extend(["col_" + str(i) for i in range(1, 61)])



    def read_us_prps_sampledata(self, filename, staname):
        # info_list = self.read_us_prps_info()[5000:5020]
        # filename = info_list["FILE_NAME"][5000]
        # staname = info_list["STATION_NAME"][5000]
        # 表名
        table_name = "us_waveform_prps_sampledata"
        # 列名
        col_names = ["id", "FILE_NAME", "STATION_NAME"]
        # 列名还有coll_1到coll_60
        col_names.extend(["col_" + str(i) for i in range(1, 61)])
        rebuild_data = []
        if self.conn.is_connected():
            cursor = self.conn.cursor()
            file_name = filename
            sta_name = staname

            # 构建查询语句
            query = f"SELECT {', '.join(col_names)} FROM {table_name} WHERE FILE_NAME = '{file_name}'"# AND SUBSTR(STATION_NAME, 1, 4) = '{sta_name}'"
            # 执行查询
            cursor.execute(query)

            # 获取查询结果
            newdata = cursor.fetchall()
            # 将datas转成ndarray
            newdata = np.array(newdata)
            newdata = newdata[:50, 3:].astype(np.float32)
            # 计算newdata总值
            pulse_count = np.sum(np.sum(newdata, axis=0))
            newdata = np.expand_dims(newdata, axis=0)  # 脉冲数量
            # 关闭cursor
            cursor.close()
        return newdata, pulse_count

    
    def __getitem__(self, index):
        # DataLoader会自动调用数据集的__getitem__方法来收集批量数据。
        features = self.tst[index]
        return features

    def __len__(self):
        return len(self.tst)


class Dataset_load_window_tst(Dataset):
    def __init__(self, map_type_code, filename, staname):
        # 连接数据库
        with DatabaseConnection() as db:
            self.conn = db
            self.cur = self.conn.cursor()
            if map_type_code == "0x35":
                self.tst, self.pulse_count = self.read_us_prpd_sampledata(filename, staname)
            elif map_type_code == "0x36":
                self.tst, self.pulse_count = self.read_us_prps_sampledata(filename, staname)

    def read_us_prpd_info(self):
        # 表名
        table_name = "us_waveform_prpd_info"
        # 列名
        col_names = ["id", "FILE_NAME", "STATION_NAME"]
        # 提取数据中col_names的数据
        self.cur.execute(f"SELECT {', '.join(col_names)} FROM {table_name}")

        rows = self.cur.fetchall()
        # 转成DataFrame
        df = pd.DataFrame(rows, columns=col_names)
        return df

    def read_us_prpd_sampledata(self, filename, staname):
        info_list = self.read_us_prpd_info()[5000:5020]
        # filename = info_list["FILE_NAME"][5000]
        # staname = info_list["STATION_NAME"][5000]
        # 表名
        table_name = "us_waveform_prpd_sampledata"
        # 列名
        col_names = ["id", "FILE_NAME", "STATION_NAME"]
        # 列名还有coll_1到coll_60
        col_names.extend(["col_" + str(i) for i in range(1, 61)])
        if self.conn.is_connected():
            cursor = self.conn.cursor()
            file_name = filename
            sta_name = staname
            # 构建查询语句
            window_data = []
            for file in filename:
                query = f"SELECT {', '.join(col_names)} FROM {table_name} WHERE FILE_NAME = '{file}'"
                # 执行查询
                cursor.execute(query)

                # 获取查询结果
                newdata = cursor.fetchall()
                # 将datas转成ndarray
                newdata = np.array(newdata)
                newdata = newdata[:80, 3:].astype(np.float32)
                # Windows数据
                window_data.append(newdata)
            # window_data 
            window_data = np.array(window_data)
            # 计算newdata总值
            pulse_count = np.sum(np.sum(window_data, axis=0)) # 脉冲数量
            window_data = np.expand_dims(window_data, axis=0)
            # 关闭cursor
            cursor.close()
        return window_data, pulse_count
    
    def read_us_prps_sampledata(self, filename, staname):
        # info_list = self.read_us_prps_info()[5000:5020]
        # filename = info_list["FILE_NAME"][5000]
        # staname = info_list["STATION_NAME"][5000]
        # 表名
        table_name = "us_waveform_prps_sampledata"
        # 列名
        col_names = ["id", "FILE_NAME", "STATION_NAME"]
        # 列名还有coll_1到coll_60
        col_names.extend(["col_" + str(i) for i in range(1, 61)])
        rebuild_data = []
        if self.conn.is_connected():
            cursor = self.conn.cursor()
            file_name = filename
            sta_name = staname

            # 构建查询语句
            window_data = []
            for file in filename:
                query = f"SELECT {', '.join(col_names)} FROM {table_name} WHERE FILE_NAME = '{file}'"
                # 执行查询
                cursor.execute(query)

                # 获取查询结果
                newdata = cursor.fetchall()
                # 将datas转成ndarray
                newdata = np.array(newdata)
                newdata = newdata[:50, 3:].astype(np.float32)
                # Windows数据
                window_data.append(newdata)
            # window_data 
            window_data = np.array(window_data)
            # 计算newdata总值
            pulse_count = np.sum(np.sum(window_data, axis=0)) # 脉冲数量
            window_data = np.expand_dims(window_data, axis=0)
            # 关闭cursor
            cursor.close()
        return window_data, pulse_count

    
    def __getitem__(self, index):
        # DataLoader会自动调用数据集的__getitem__方法来收集批量数据。
        features = self.tst[index]
        return features

    def __len__(self):
        return len(self.tst)