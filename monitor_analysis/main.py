# 实际运行程序
# 导入库
import sys

sys.path.append("..")
import numpy as np
import pandas as pd
import argparse
import torch
from Data_loader_tst import Dataset_load_tst, Dataset_load_window_tst
from model_architecture import CustomNet, CustomWinNet
from torch.utils.data import DataLoader
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import math


def get_args_parser():
    parser = argparse.ArgumentParser("Partial Discharge Model training", add_help=False)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--detail_step", default=1, type=int)
    parser.add_argument("--map_type_code", default="0x35", type=str)  # 图谱类型编码

    # Model parameters
    parser.add_argument("--model_name", default="pd", type=str)
    parser.add_argument("--finetune", default=False, type=bool)
    parser.add_argument("--checkpoint", default="RUL_Transformer_mask.pth", type=str)

    # Optimizer parameters
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--lr", type=float, default=1e-3, metavar="LR")
    # 模型保存地址
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="D://Graduate/projects/partial_discharge_monitoring_20230904/AI-project-envs/saved_models/",
    )
    return parser


def pred(args, model, tst_dataset, filename, staname):
    # 创建tst_loader
    tst_loader = DataLoader(tst_dataset, batch_size=1, shuffle=False)
    # 导入模型参数
    model.load_state_dict(
        torch.load(
            args.model_save_path
            + "pd_model_{}_{}.pth".format(args.map_type_code, args.mode),
            map_location=torch.device("cpu"),
        )
    )
    # 预测结果
    model.eval()
    with torch.no_grad():
        for inputs in tst_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            prob, predicted = torch.max(outputs.data, 1)
            # predictions.extend(predicted.cpu().numpy().tolist())
        # 打印概率与预测结果
        print("图谱类型：{}".format(args.map_type_code))
        print(
            "当前放电类型：{}, 当前放电概率：{}".format(predicted.item(), prob.item())
        )
    # save the prediction result to the database
    save_pred_result(prob, predicted, filename, staname, args.map_type_code, args.mode)


def main(args):

    # prpd data

    # get data_list that newly inserted into the database,if there is no new data, the program will stop
    info_list_prpd = read_us_data_info()
    info_list_prps = read_us_data_info(table_name="us_waveform_prps_info_bak")
    info_list = pd.merge(
        info_list_prpd, info_list_prps, on=["FILE_NAME", "STATION_NAME"], how="inner"
    )

    if len(info_list) == 0:
        print("This iteration has no new data.Please wait for the next iteration.")
        return None

    # model preparation
    model_prpd_s = CustomNet(80, 60)
    model_prpd_s.to(device)
    model_prps_s = CustomNet(50, 60)
    model_prps_s.to(device)

    model_prpd_w = CustomWinNet(80, 60)
    model_prpd_w.to(device)
    model_prps_w = CustomWinNet(50, 60)
    model_prps_w.to(device)

    # for loop,every item in the data_list will be used to predict,and the result will be saved to the database and the last_id will be updated
    for i in range(len(info_list)):
        filename = info_list["FILE_NAME"][i]
        staname = info_list["STATION_NAME"][i]

        filenames = get_filenames(info_list["FILE_NAME"], i)

        # prpd data
        args.map_type_code = "0x35"
        prpd_dataset_s = Dataset_load_tst(args.map_type_code, filename, staname)
        if len(filenames) != 0:
            prpd_dataset_w = Dataset_load_window_tst(
                args.map_type_code, filenames, staname
            )

        # prps data
        args.map_type_code = "0x36"
        prps_dataset_s = Dataset_load_tst(args.map_type_code, filename, staname)
        if len(filenames) != 0:
            prps_dataset_w = Dataset_load_window_tst(
                args.map_type_code, filenames, staname
            )

        # prediction
        # the pred func will predict the result and save the result to the database,and every operation will update the file name and station number

        # single data
        args.mode = "s"
        print("单图谱预测")
        args.map_type_code = "0x35"
        pred(args, model_prpd_s, prpd_dataset_s, filename, staname)
        args.map_type_code = "0x36"
        pred(args, model_prps_s, prps_dataset_s, filename, staname)

        # window data
        args.mode = "w"
        print("窗口图谱预测")
        if len(filenames) !=0:
            args.map_type_code = "0x35"
            pred(args, model_prpd_w, prpd_dataset_w, filename, staname)
            args.map_type_code = "0x36"
            pred(args, model_prps_w, prps_dataset_w, filename, staname)
        else:
            print('新文件不足三个，无法进行窗口图谱预测')


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
