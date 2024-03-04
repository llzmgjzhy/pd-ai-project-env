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


def pred(args, model, tst_dataset, filename, stanam, id):
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
        save_pred_result(prob, predicted, filename, stanam, args.map_type_code, id)


def main(args):

    # get data_list that newly inserted into the database,if there is no new data, the program will stop
    info_list, latest_id = read_us_prpd_info()
    if info_list is None:
        print("This iteration has no new data.Please wait for the next iteration.")
        return None

    # model preparation
    if args.map_type_code == "0x35":
        model = CustomNet(80, 60)
    elif args.map_type_code == "0x36":
        model = CustomNet(50, 60)
    model.to(device)

    # for loop,every item in the data_list will be used to predict,and the result will be saved to the database and the last_id will be updated
    for i in range(len(info_list)):
        id = info_list["id"][i]
        filename = info_list["FILE_NAME"][i]
        staname = info_list["STATION_NAME"][i]
        test_dataset = Dataset_load_tst(args.map_type_code, filename, staname)
        args.mode = "s"
        print("单图谱预测")
        # the pred func will predict the result and save the result to the database,and every operation will update the last_id
        pred(args, model, test_dataset, filename, staname, id)

    # TODO now we just test prpd data,we will add prps data later

    # if args.map_type_code == "0x35":
    #     # 构建网络模型
    #     model = CustomNet(80, 60)
    # elif args.map_type_code == "0x36":
    #     model = CustomNet(50, 60)
    # model.to(device)
    # test_dataset = Dataset_load_tst(args.map_type_code, filename, staname)
    # args.mode = "s" # 单图谱
    # print("单图谱预测")
    # # 预测
    # pred(args, model, test_dataset)

    # filename = ['AA_20240226225000000.dat', 'AA_20240226225500000.dat','AA_20240226230000000.dat']
    # # 构造窗口数据集模型
    # if args.map_type_code == "0x35":
    #     # 构建网络模型
    #     model = CustomWinNet(80, 60)
    # elif args.map_type_code == "0x36":
    #     model = CustomWinNet(50, 60)
    #     # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # model.to(device)
    # # 读取实际运行数据
    # test_dataset = Dataset_load_window_tst(args.map_type_code, filename, staname)
    # args.mode = "w"  # 窗口图谱
    # print("窗口图谱预测")
    # pred(args, model, test_dataset)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    args.map_type_code = "0x35"
    main(args)
