from mysql_db import DatabaseConnection
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from abc import ABC
from abc import abstractmethod

# base class,define the common method
# including read_info and read_data,which are abstract method
class Base_DataLoader_Tst(ABC):
    def __init__(self,info_table_name,data_table_name) -> None:
        self.info_table_name = info_table_name
        self.data_table_name = data_table_name
        self.tst = []

    @abstractmethod
    def read_info(self):
        """
        read the info data according to the info table
        """
        pass

    @abstractmethod
    def read_data(self):
        """
        read the data according to the data table
        """
        pass

    def __getitem__(self, index):
        features = self.tst[index]
        return features

    def __len__(self):
        return len(self.tst)


class Dataset_load_tst(Dataset):
    def __init__(self, map_type_code, filename, staname, id=0):
        # 连接数据库
        with DatabaseConnection() as db:
            self.conn = db
            self.cur = self.conn.cursor()
            if map_type_code == "0x35":
                self.tst, self.pulse_count = self.read_us_prpd_sampledata(
                    filename, staname
                )
            elif map_type_code == "0x36":
                self.tst, self.pulse_count = self.read_us_prps_sampledata(
                    filename, staname
                )
            elif map_type_code == "0x31":
                self.tst, self.file_info = self.read_us_feature_info(
                    filename, staname, id
                )

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

    def read_us_feature_info(self, filename, staname, id):

        mea_pos_name = [
            "614开关柜",
            "615开关柜",
            "616开关柜",
            "620开关柜",
            "620电缆仓上",
            "620电缆仓下",
        ]
        table_name = "us_features_info"
        self.cur.execute(f"SELECT * FROM {table_name} WHERE id > '{id}'")

        origin_data = self.cur.fetchall()
        col_names = [desc[0] for desc in self.cur.description]

        # 转成DataFrame
        df = pd.DataFrame(origin_data, columns=col_names)

        for i in range(len(mea_pos_name)):
            selected_data = (
                df[df["measure_position_name"].str.contains(mea_pos_name[i])]
                .iloc[:, [0] + list(range(19, 23))]
                .astype(float)
                .values
            )
            selected_data_file = (
                df[df["measure_position_name"].str.contains(mea_pos_name[i])]
                .iloc[:, [0, 1, 8, 9]]
                .values
            )
            if i == 0:
                data = selected_data
                file_info_data = selected_data_file
            else:
                data = np.concatenate((data, selected_data), axis=0)
                file_info_data = np.concatenate(
                    (file_info_data, selected_data_file), axis=0
                )
        # data的第一列，第二列最大值73，最小值是-7，归一化处理
        min_vol = -7
        max_vol = 73
        data[:, 1] = (data[:, 1] - min_vol) / (max_vol - min_vol)
        data[:, 2] = (data[:, 2] - min_vol) / (max_vol - min_vol)
        data = data[data[:, 0].argsort()]
        data = data[:, 1:]
        file_info_data = file_info_data[file_info_data[:, 0].argsort()]
        return data, file_info_data

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
            query = f"SELECT {', '.join(col_names)} FROM {table_name} WHERE FILE_NAME = '{file_name}' AND STATION_NAME = '{sta_name}'"  # AND SUBSTR(STATION_NAME, 1, 4) = '{sta_name}'"
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

    def __getitem__(self, index):
        # DataLoader会自动调用数据集的__getitem__方法来收集批量数据。
        features = self.tst[index]
        return features

    def __len__(self):
        return len(self.tst)


class Dataset_load_window_tst(Dataset):
    def __init__(self, map_type_code, filename, staname, id=0):
        # 连接数据库
        with DatabaseConnection() as db:
            self.conn = db
            self.cur = self.conn.cursor()
            if map_type_code == "0x35":
                self.tst, self.pulse_count = self.read_us_prpd_sampledata(
                    filename, staname
                )
            elif map_type_code == "0x36":
                self.tst, self.pulse_count = self.read_us_prps_sampledata(
                    filename, staname
                )
            elif map_type_code == "0x31":
                self.tst, self.file_info = self.read_us_feature_info(
                    filename, staname, id
                )

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

    def read_us_feature_info(self, filename, staname, id, window_size=3):
        mea_pos_name = [
            "614开关柜",
            "615开关柜",
            "616开关柜",
            "620开关柜",
            "620电缆仓上",
            "620电缆仓下",
        ]
        table_name = "us_features_info"
        self.cur.execute(f"SELECT * FROM {table_name} WHERE id > '{id}'")

        origin_data = self.cur.fetchall()
        col_names = [desc[0] for desc in self.cur.description]

        # 转成DataFrame
        df = pd.DataFrame(origin_data, columns=col_names)

        for i in range(len(mea_pos_name)):
            selected_data = (
                df[df["measure_position_name"].str.contains(mea_pos_name[i])]
                .iloc[:, [0] + list(range(19, 23))]
                .astype(float)
                .values
            )
            selected_data_file = (
                df[df["measure_position_name"].str.contains(mea_pos_name[i])]
                .iloc[:, [0, 1, 8, 9]]
                .values
            )
            if i == 0:
                data = selected_data
                file_info_data = selected_data_file
            else:
                data = np.concatenate((data, selected_data), axis=0)
                file_info_data = np.concatenate(
                    (file_info_data, selected_data_file), axis=0
                )
        # data的第一列，第二列最大值73，最小值是-7，归一化处理
        min_vol = -7
        max_vol = 73
        data[:, 1] = (data[:, 1] - min_vol) / (max_vol - min_vol)
        data[:, 2] = (data[:, 2] - min_vol) / (max_vol - min_vol)

        data = data[data[:, 0].argsort()]
        data = data[:, 1:]
        file_info_data = file_info_data[file_info_data[:, 0].argsort()]

        # 生成窗口数据
        window_data = []
        for i in range(data.shape[0]):
            if i + window_size > data.shape[0]:
                window_data.append(data[i - window_size : i, :])
            else:
                window_data.append(data[i : i + window_size, :])
        window_data = np.array(window_data)
        return window_data, file_info_data

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
        if self.conn.is_connected():
            cursor = self.conn.cursor()
            file_name = filename
            sta_name = staname
            # 构建查询语句
            window_data = []
            for file in file_name:
                query = f"SELECT {', '.join(col_names)} FROM {table_name} WHERE FILE_NAME = '{file}' AND STATION_NAME = '{sta_name}'"
                # 执行查询
                cursor.execute(query)

                # 获取查询结果
                newdata = cursor.fetchall()
                # 将datas转成ndarray
                newdata = np.array(newdata)
                newdata = newdata[:, 3:].astype(np.float32)
                # Windows数据
                window_data.append(newdata)
            # window_data
            window_data = np.array(window_data)
            # 计算newdata总值
            pulse_count = np.sum(np.sum(window_data, axis=0))  # 脉冲数量
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
            for file in file_name:
                query = f"SELECT {', '.join(col_names)} FROM {table_name} WHERE FILE_NAME = '{file}' AND STATION_NAME = '{sta_name}'"
                # 执行查询
                cursor.execute(query)

                # 获取查询结果
                newdata = cursor.fetchall()
                # 将datas转成ndarray
                newdata = np.array(newdata)
                newdata = newdata[:, 3:].astype(np.float32)
                # Windows数据
                window_data.append(newdata)
            # window_data
            window_data = np.array(window_data)
            # 计算newdata总值
            pulse_count = np.sum(np.sum(window_data, axis=0))  # 脉冲数量
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
