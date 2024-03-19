# 实际运行程序
# 导入库
import sys

sys.path.append("..")
import numpy as np
import pandas as pd
import argparse
import torch
from Data_loader_tst import Dataset_load_tst, Dataset_load_window_tst
from model_architecture import VoltageWinNet, VoltageNet
from torch.utils.data import DataLoader
from utils import *
import re
import os
from pathlib import Path

cur_dir = Path(__file__).resolve().parent.parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import math


def get_args_parser():
    parser = argparse.ArgumentParser("Partial Discharge Model Running", add_help=False)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--detail_step", default=1, type=int)
    parser.add_argument("--map_type_code", default="0x31", type=str)  # 图谱类型编码

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
        default=f"{cur_dir}/saved_models/",
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
            # 应用阈值，将小于0.5的标签设为0，大于等于0.5的标签设为
            predicted = (outputs >= 0.5).float()
            prob = torch.sigmoid(outputs)
            # predictions.extend(predicted.cpu().numpy().tolist())
        # 打印概率与预测结果
        print("图谱类型：{}".format(args.map_type_code))
        print(
            "当前放电类型：{}, 当前放电概率：{}".format(predicted.item(), prob.item())
        )
    # save the prediction result to the database
    save_pred_result(prob, predicted, filename, staname, args.map_type_code, args.mode)


def test(args, model, test_data, filename, pos_name, pos_code,id = 0):
    # 设置模型为评估模式
    input = torch.tensor(test_data).to(device)
    model = model.to(device)

    model.eval()
    # test_labels = torch.tensor(test_labels).to(device)

    # 使用无梯度计算上下文管理器
    with torch.no_grad():
        # 将测试数据输入模型进行预测
        inputs = input.to(torch.float32)
        # inputs = test_data
        outputs = model(inputs)
        # 计算准确率
        predicted_labels = torch.round(outputs)
        prob = torch.sigmoid(outputs)
        print("图谱类型：{}".format(args.map_type_code))
        print("当前放电类型：{}, 当前放电概率：{}".format(predicted_labels, prob))

    # 打印测试结果
    save_pred_result_voltage(
        prob,
        predicted_labels,
        filename,
        pos_name,
        pos_code,
        args.map_type_code,
        args.mode,
        id
    )


def main(args):

    args.map_type_code = "0x31"
    args.mode = "s"  # 单图谱

    model_s = VoltageNet(4, 1)
    model_s.load_state_dict(
        torch.load(
            args.model_save_path
            + "pd_model_{}_{}.pth".format(args.map_type_code, args.mode),
            map_location=torch.device("cpu"),
        )
    )

    args.mode = "w"  # 窗口图谱
    model_w = VoltageWinNet(4, 1)
    model_w.load_state_dict(
        torch.load(
            args.model_save_path
            + "pd_model_{}_{}.pth".format(args.map_type_code, args.mode),
            map_location=torch.device("cpu"),
        )
    )

    filename, staname, id = get_last_file_wireless()

    # 单窗口数据集
    dataset_s = Dataset_load_tst(
        map_type_code=args.map_type_code, filename=filename, staname=staname, id=id
    )
    file_info_s = dataset_s.file_info
    X_test_s = dataset_s.tst

    # 多窗口数据集
    dataset_w = Dataset_load_window_tst(
        map_type_code=args.map_type_code, filename=filename, staname=staname, id=id
    )
    file_info_w = dataset_w.file_info
    X_test_w = dataset_w.tst

    # args.mode = "s"
    # for i in range(len(file_info_s)):
    #     id = file_info_s[i][0]
    #     filename = file_info_s[i][1]
    #     pos_name = file_info_s[i][2]
    #     pos_code = file_info_s[i][3]

    #     test(args, model_s, X_test_s[i], filename, pos_name, pos_code)

    args.mode = "w"
    for i in range(len(file_info_w)):
        id = file_info_w[i][0]
        filename = file_info_w[i][1]
        pos_name = file_info_w[i][2]
        pos_code = file_info_w[i][3]

        test(args, model_w, X_test_w[i], filename, pos_name, pos_code,id)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
