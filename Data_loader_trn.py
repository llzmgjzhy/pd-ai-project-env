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
            elif map_type_code == "0x31":
                self.trn, self.trn_y, self.file_info = self.read_us_feature_info()

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

    def read_us_feature_info(self):
        # 表名
        table_name = "us_features_info"
        # 列名
        # col_names = ["id", "FILE_NAME", "STATION_NAME"]
        mea_pos_code = [
            "B1238001000000000000000000000000",
            "B1238002000000000000000000000000",
            "B3226002000000000000000000000000",
            "B1238004000000000000000000000000",
            "B1238003000000000000000000000000",
            "B3226001000000000000000000000000",
        ]
        mea_pos_name = [
            "620电缆仓上",
            "620电缆仓下",
            "614开关柜",
            "615开关柜",
            "616开关柜",
            " 620开关柜",
        ]
        y = [1, 1, 1, 0, 0, 0]
        # 620电缆仓上，620电缆仓下, 614开关柜, 615开关柜, 616开关柜, 620开关柜
        # 提取数据中列名为measure_position_code值为mea_pos_code的所有数据
        self.cur.execute(
            f"SELECT * FROM {table_name} WHERE measure_position_code in {tuple(mea_pos_code)}"
        )

        rows = self.cur.fetchall()
        # 得到所有列名
        col_names = [desc[0] for desc in self.cur.description]

        # 转成DataFrame
        df = pd.DataFrame(rows, columns=col_names)
        file_info = df[["file_name", "measure_position_name", "measure_position_code"]]
        for i in range(len(mea_pos_code)):
            # d = df['measure_position_name'][:3]
            selected_data = (
                df[df["measure_position_name"].str.contains(mea_pos_name[i])]
                .iloc[:, 19:23]
                .astype(float)
                .values
            )
            selected_data_file = (
                df[df["measure_position_name"].str.contains(mea_pos_name[i])]
                .iloc[:, [1, 8, 9]]
                .values
            )
            # 制作标签

            if i == 0:
                data = selected_data
                file_info_data = selected_data_file
                label = np.ones((selected_data.shape[0], 1)) * y[i]
            else:
                data = np.concatenate((data, selected_data), axis=0)
                file_info_data = np.concatenate((file_info_data, selected_data_file), axis=0)
                label = np.concatenate(
                    (label, np.ones((selected_data.shape[0], 1)) * y[i]), axis=0
                )
        # data的第一列，第二列最大值73，最小值是-7，归一化处理
        min_vol = -7
        max_vol = 73
        data[:, 0] = (data[:, 0] - min_vol) / (max_vol - min_vol)
        data[:, 1] = (data[:, 1] - min_vol) / (max_vol - min_vol)

        return data, label, file_info_data

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
                    y_label.append(float(1))
                    sta_name = "614开关柜"
                else:
                    y_label.append(float(0))
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
        info_list = self.read_us_prps_info()
        # filename = info_list["FILE_NAME"][5000:5020]
        # staname = info_list["STATION_NAME"][5000:5020]
        # 表名
        table_name = "us_waveform_prps_sampledata"
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
                device_id = re.findall(r"\d+", sta_name)[0]
                if device_id == "614":
                    y_label.append(float(1))
                    sta_name = "614开关柜"
                else:
                    y_label.append(float(0))
                    sta_name = "620开关柜"
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
            self.y_label = np.array(y_label)
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
            elif map_type_code == "0x31":
                self.trn, self.trn_y, self.file_info = self.read_us_feature_info()

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

    def read_us_feature_info(self, window_size=3):
        # 表名
        table_name = "us_features_info"
        # 列名
        mea_pos_code = [
            "B1238001000000000000000000000000",
            "B1238002000000000000000000000000",
            "B3226002000000000000000000000000",
            "B1238004000000000000000000000000",
            "B1238003000000000000000000000000",
            "B3226001000000000000000000000000",
        ]
        mea_pos_name = [
            "620电缆仓上",
            "620电缆仓下",
            "614开关柜",
            "615开关柜",
            "616开关柜",
            "620开关柜",
        ]
        y = [1, 1, 1, 0, 0, 0]

        # 提取数据中列名为measure_position_code值为mea_pos_code的所有数据
        self.cur.execute(
            f"SELECT * FROM {table_name} WHERE measure_position_code in {tuple(mea_pos_code)}"
        )

        rows = self.cur.fetchall()
        # 得到所有列名
        col_names = [desc[0] for desc in self.cur.description]

        # 转成DataFrame
        df = pd.DataFrame(rows, columns=col_names)
        file_info = df[["FILE_NAME", "measure_position_name", "measure_position_code"]]

        # 初始化空列表用于存储窗口数据和标签
        window_data = []
        window_label = []

        for i in range(len(mea_pos_code)):
            selected_data = (
                df[df["measure_position_name"].str.contains(mea_pos_name[i])]
                .iloc[:, 19:23]
                .astype(float)
                .values
            )

            # 制作标签
            label = np.ones((selected_data.shape[0], 1)) * y[i]

            # 将数据划分为多个窗口
            num_windows = len(selected_data) - window_size + 1
            for j in range(num_windows):
                window = selected_data[j : j + window_size]
                window_data.append(window)
                window_label.append(y[i])

        # 将列表转换为NumPy数组
        window_data = np.array(window_data)
        window_label = np.array(window_label)

        # 归一化处理
        min_vol = -7
        max_vol = 73
        window_data[:, :, 0] = (window_data[:, :, 0] - min_vol) / (max_vol - min_vol)
        window_data[:, :, 1] = (window_data[:, :, 1] - min_vol) / (max_vol - min_vol)

        return window_data, window_label, file_info

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

        sub_1_info_list = info_list[
            (info_list["STATION_NAME"] == "614开关柜")
        ].reset_index(drop=True)
        sub_2_info_list = info_list[
            (info_list["STATION_NAME"] == "620开关柜")
        ].reset_index(drop=True)

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
                file_name = get_filenames(sub_1_info_list["FILE_NAME"], i)
                sta_name = sub_1_info_list["STATION_NAME"].iloc[i]
                device_id = re.findall(r"\d+", sta_name)[0]
                if device_id == "614":
                    y_label.append(float(1))
                    sta_name = "614开关柜"
                else:
                    y_label.append(float(0))
                    sta_name = "620开关柜"
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
                    # 将newdata添加到rebuild_data
                    window_data.append(newdata)
                # 将newdata添加到rebuild_data
                rebuild_data.append(window_data)
            for i in range(sub_2_info_list.shape[0]):
                file_name = get_filenames(sub_2_info_list["FILE_NAME"], i)
                # 构建查询语句
                window_data = []

                sta_name = sub_2_info_list["STATION_NAME"].iloc[i]
                device_id = re.findall(r"\d+", sta_name)[0]
                if device_id == "614":
                    y_label.append(float(1))
                    sta_name = "614开关柜"
                else:
                    y_label.append(float(0))
                    sta_name = "620开关柜"
                for file in file_name:
                    query = f"SELECT {', '.join(col_names)} FROM {table_name} WHERE FILE_NAME = '{file}' AND STATION_NAME = '{sta_name}'"
                    # 执行查询
                    cursor.execute(query)
                    # 获取查询结果
                    newdata = cursor.fetchall()
                    # 将datas转成ndarray
                    newdata = np.array(newdata)
                    newdata = newdata[:, 3:].astype(np.float32)
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
        info_list = self.read_us_prps_info()
        sub_1_info_list = info_list[
            (info_list["STATION_NAME"] == "614开关柜")
        ].reset_index(drop=True)
        sub_2_info_list = info_list[
            (info_list["STATION_NAME"] == "620开关柜")
        ].reset_index(drop=True)
        # filename = info_list["FILE_NAME"][5000:5020]
        # staname = info_list["STATION_NAME"][5000:5020]
        # 表名
        table_name = "us_waveform_prps_sampledata"
        # 列名
        col_names = ["id", "FILE_NAME", "STATION_NAME"]
        # 列名还有coll_1到coll_60
        col_names.extend(["col_" + str(i) for i in range(1, 61)])
        rebuild_data = []
        y_label = []
        if self.conn.is_connected():
            cursor = self.conn.cursor()
            for i in range(sub_1_info_list.shape[0]):
                file_name = get_filenames(sub_1_info_list["FILE_NAME"], i)
                sta_name = sub_1_info_list["STATION_NAME"].iloc[i]
                device_id = re.findall(r"\d+", sta_name)[0]
                if device_id == "614":
                    y_label.append(float(1))
                    sta_name = "614开关柜"
                else:
                    y_label.append(float(0))
                    sta_name = "620开关柜"
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
                    # 将newdata添加到rebuild_data
                    window_data.append(newdata)
                # 将window_data添加到rebuild_data
                rebuild_data.append(window_data)
            for i in range(sub_2_info_list.shape[0]):
                file_name = get_filenames(sub_2_info_list["FILE_NAME"], i)
                sta_name = sub_2_info_list["STATION_NAME"].iloc[i]
                device_id = re.findall(r"\d+", sta_name)[0]
                if device_id == "614":
                    y_label.append(float(1))
                    sta_name = "614开关柜"
                else:
                    y_label.append(float(0))
                    sta_name = "620开关柜"
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
                    # 将newdata添加到rebuild_data
                    window_data.append(newdata)
                # 将window_data添加到rebuild_data
                rebuild_data.append(window_data)
            # rebuild_data转成ndarray
            self.y_label = np.array(y_label)
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
