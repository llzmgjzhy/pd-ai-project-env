from Mysql_connect import DatabaseConnection
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import re
from monitor_analysis.utils import get_filenames


class Dataset_load_trn(Dataset):
    def __init__(self, map_type_code, train=True):
        self.train = train
        self.map_type_code = map_type_code
        # 连接数据库
        with DatabaseConnection() as db:
            self.conn = db
            self.cur = self.conn.cursor()
            if map_type_code == "0x35":
                self.trn = self.read_us_prpd_sampledata()
                self.trn_y = self.generate_y_data()
            elif map_type_code == "0x36":
                self.trn = self.read_us_prps_sampledata()
                self.trn_y = self.generate_y_data()

        # 划分训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.trn, self.trn_y, test_size=0.2, random_state=42
        )

    def generate_y_data(self):
        # y_data = []
        # for row in self.trn:
        # 生成随机的两列数据，使得它们的和为1
        # rand_val = np.random.rand(2)
        # while rand_val[0] <= rand_val[1]:
        #     rand_val = np.random.rand(2)
        # rand_val /= np.sum(rand_val)
        # y_data.append(rand_val)
        y_data = self.y_label
        return np.array(y_data)

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

    def read_us_prpd_sampledata(self):
        info_list = self.read_us_prpd_info()
        # filename = info_list["FILE_NAME"][5000:5020]
        # staname = info_list["STATION_NAME"][5000:5020]
        # 表名
        table_name = "us_waveform_prpd_sampledata"
        # 列名
        col_names = ["id", "FILE_NAME", "STATION_NAME"]
        # 列名还有coll_1到coll_60
        col_names.extend(["col_" + str(i) for i in range(1, 61)])
        rebuild_data = []
        y_label = []
        if self.conn.is_connected():
            cursor = self.conn.cursor()
            for i in range(info_list.shape[0]):
                file_name = info_list["FILE_NAME"].iloc[i]
                sta_name = info_list["STATION_NAME"].iloc[i]
                # 构建查询语句
                device_id = re.findall(r"\d+", sta_name)[0]
                if device_id == "614":
                    y_label.append(np.array([0.9,0.1]))
                    sta_name = "614开关柜"
                else:
                    y_label.append(np.array([0.1,0.9]))
                    sta_name = "620开关柜"
                query = f"SELECT {', '.join(col_names)} FROM {table_name} WHERE FILE_NAME = '{file_name}' AND STATION_NAME = '{sta_name}'"
                # 执行查询
                cursor.execute(query)
                # 获取查询结果
                newdata = cursor.fetchall()
                # 将datas转成ndarray
                newdata = np.array(newdata)
                newdata = newdata[:, 3:].astype(np.float32)
                # 将newdata添加到rebuild_data
                rebuild_data.append(newdata)
            # rebuild_data转成ndarray
            rebuild_data = np.array(rebuild_data)
            self.y_label = np.array(y_label)
            # 关闭cursor
            cursor.close()
        return rebuild_data

    def read_us_prps_info(self):
        # 表名
        table_name = "us_waveform_prps_info"
        # 列名
        col_names = ["id", "FILE_NAME", "STATION_NAME"]
        # 提取数据中col_names的数据
        self.cur.execute(f"SELECT {', '.join(col_names)} FROM {table_name}")

        rows = self.cur.fetchall()
        # 转成DataFrame
        df = pd.DataFrame(rows, columns=col_names)
        return df

    def read_us_prps_sampledata(self):
        info_list = self.read_us_prps_info()[5000:5020]
        # filename = info_list["FILE_NAME"][5000:5020]
        # staname = info_list["STATION_NAME"][5000:5020]
        # 表名
        table_name = "us_waveform_prps_sampledata"
        # 列名
        col_names = ["id", "FILE_NAME", "STATION_NAME"]
        # 列名还有coll_1到coll_60
        col_names.extend(["col_" + str(i) for i in range(1, 61)])
        rebuild_data = []
        if self.conn.is_connected():
            cursor = self.conn.cursor()
            for i in range(info_list.shape[0]):
                file_name = info_list["FILE_NAME"].iloc[i]
                sta_name = info_list["STATION_NAME"].iloc[i]
                # 构建查询语句
                query = f"SELECT {', '.join(col_names)} FROM {table_name} WHERE FILE_NAME = '{file_name}' AND STATION_NAME = '{sta_name}'"
                # 执行查询
                cursor.execute(query)
                # 获取查询结果
                newdata = cursor.fetchall()
                # 将datas转成ndarray
                newdata = np.array(newdata)
                newdata = newdata[:, 3:].astype(np.float32)
                # 将newdata添加到rebuild_data
                rebuild_data.append(newdata)
            # rebuild_data转成ndarray
            rebuild_data = np.array(rebuild_data)
            # 关闭cursor
            cursor.close()
        return rebuild_data

    def __getitem__(self, index):
        # DataLoader会自动调用数据集的__getitem__方法来收集批量数据。
        if self.train:
            features, labels = self.X_train[index], self.y_train[index]
        else:
            features, labels = self.X_test[index], self.y_test[index]
        return features, labels

    def __len__(self):
        if self.train:
            return len(self.X_train)
        else:
            return len(self.X_test)


class Dataset_load_window_trn(Dataset):
    def __init__(self, map_type_code, train=True, winsow_size=3):
        self.train = train
        self.map_type_code = map_type_code
        self.window_size = winsow_size
        # 连接数据库
        with DatabaseConnection() as db:
            self.conn = db
            self.cur = self.conn.cursor()
            if map_type_code == "0x35":
                self.trn = self.read_us_prpd_sampledata()
                self.trn_y = self.generate_y_data()
            elif map_type_code == "0x36":
                self.trn = self.read_us_prps_sampledata()
                self.trn_y = self.generate_y_data()

        # 划分训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.trn, self.trn_y, test_size=0.2, random_state=42
        )

    def generate_y_data(self):
        # y_data = []
        # for row in self.trn:
        #     # 生成随机的两列数据，使得它们的和为1
        #     rand_val = np.random.rand(2)
        #     while rand_val[0] <= rand_val[1]:
        #         rand_val = np.random.rand(2)
        #     rand_val /= np.sum(rand_val)
        #     y_data.append(rand_val)

        y_data = self.y_label
        return np.array(y_data)

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

    def read_us_prpd_sampledata(self):
        info_list = self.read_us_prpd_info()
        
        sub_1_info_list = info_list[(info_list["STATION_NAME"] == "614开关柜")].reset_index(drop=True)
        sub_2_info_list = info_list[(info_list["STATION_NAME"] == "620开关柜")].reset_index(drop=True)

        # filename = info_list["FILE_NAME"][5000:5020]
        # staname = info_list["STATION_NAME"][5000:5020]
        # 表名
        table_name = "us_waveform_prpd_sampledata"
        # 列名
        col_names = ["id", "FILE_NAME", "STATION_NAME"]
        # 列名还有coll_1到coll_60
        col_names.extend(["col_" + str(i) for i in range(1, 61)])
        rebuild_data = []
        y_label = []
        if self.conn.is_connected():
            cursor = self.conn.cursor()
            for i in range(sub_1_info_list.shape[0]):
                file_name = get_filenames(sub_1_info_list['FILE_NAME'], i)
                sta_name = sub_1_info_list["STATION_NAME"].iloc[i]
                device_id = re.findall(r"\d+", sta_name)[0]
                if device_id == "614":
                    y_label.append(np.array([0.9,0.1]))
                else:
                    y_label.append(np.array([0.1,0.9]))
                # 构建查询语句
                window_data = []
                for file in file_name:
                    query = f"SELECT {', '.join(col_names)} FROM {table_name} WHERE FILE_NAME = '{file}'"
                    # 执行查询
                    cursor.execute(query)
                    # 获取查询结果
                    newdata = cursor.fetchall()
                    # 将datas转成ndarray
                    newdata = np.array(newdata)
                    newdata = newdata[:80, 3:].astype(np.float32)
                    # 将newdata添加到rebuild_data
                    window_data.append(newdata)
                # 将newdata添加到rebuild_data
                rebuild_data.append(window_data)
            for i in range(sub_2_info_list.shape[0]):
                file_name = get_filenames(sub_2_info_list['FILE_NAME'], i)
                # 构建查询语句
                window_data = []

                sta_name = sub_2_info_list["STATION_NAME"].iloc[i]
                device_id = re.findall(r"\d+", sta_name)[0]
                if device_id == "614":
                    y_label.append(np.array([0.9, 0.1]))
                else:
                    y_label.append(np.array([0.1, 0.9]))
                for file in file_name:
                    query = f"SELECT {', '.join(col_names)} FROM {table_name} WHERE FILE_NAME = '{file}'"
                    # 执行查询
                    cursor.execute(query)
                    # 获取查询结果
                    newdata = cursor.fetchall()
                    # 将datas转成ndarray
                    newdata = np.array(newdata)
                    newdata = newdata[:80, 3:].astype(np.float32)
                    # 将newdata添加到rebuild_data
                    window_data.append(newdata)
                # 将newdata添加到rebuild_data
                rebuild_data.append(window_data)
            # rebuild_data转成ndarray
            self.y_label = np.array(y_label)
            rebuild_data = np.array(rebuild_data)
            # 关闭cursor
            cursor.close()
        return rebuild_data

    def read_us_prps_info(self):
        # 表名
        table_name = "us_waveform_prps_info"
        # 列名
        col_names = ["id", "FILE_NAME", "STATION_NAME"]
        # 提取数据中col_names的数据
        self.cur.execute(f"SELECT {', '.join(col_names)} FROM {table_name}")

        rows = self.cur.fetchall()
        # 转成DataFrame
        df = pd.DataFrame(rows, columns=col_names)
        return df

    def read_us_prps_sampledata(self):
        info_list = self.read_us_prps_info()[5000:5020]
        # filename = info_list["FILE_NAME"][5000:5020]
        # staname = info_list["STATION_NAME"][5000:5020]
        # 表名
        table_name = "us_waveform_prps_sampledata"
        # 列名
        col_names = ["id", "FILE_NAME", "STATION_NAME"]
        # 列名还有coll_1到coll_60
        col_names.extend(["col_" + str(i) for i in range(1, 61)])
        rebuild_data = []
        if self.conn.is_connected():
            cursor = self.conn.cursor()
            for i in range(self.window_size, info_list.shape[0]):
                file_name = info_list["FILE_NAME"].iloc[i - 3 : i]
                sta_name = info_list["STATION_NAME"].iloc[i - 3 : i]
                # 构建查询语句
                window_data = []
                for file in file_name:
                    query = f"SELECT {', '.join(col_names)} FROM {table_name} WHERE FILE_NAME = '{file}'"
                    # 执行查询
                    cursor.execute(query)
                    # 获取查询结果
                    newdata = cursor.fetchall()
                    # 将datas转成ndarray
                    newdata = np.array(newdata)
                    newdata = newdata[:50, 3:].astype(np.float32)
                    # 将newdata添加到rebuild_data
                    window_data.append(newdata)
                # 将window_data添加到rebuild_data
                rebuild_data.append(window_data)
            # rebuild_data转成ndarray
            rebuild_data = np.array(rebuild_data)
            # 关闭cursor
            cursor.close()
        return rebuild_data

    def __getitem__(self, index):
        # DataLoader会自动调用数据集的__getitem__方法来收集批量数据。
        if self.train:
            features, labels = self.X_train[index], self.y_train[index]
        else:
            features, labels = self.X_test[index], self.y_test[index]
        return features, labels

    def __len__(self):
        if self.train:
            return len(self.X_train)
        else:
            return len(self.X_test)
