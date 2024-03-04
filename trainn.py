# 模型训练程序
# 导入库
import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn as nn
from Data_loader_trn import Dataset_load_trn, Dataset_load_window_trn
from model_architecture import CustomNet, CustomWinNet
from torch import optim
from torch.utils.data import DataLoader
from torchinfo import summary

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
    parser.add_argument("--mode", default="s", type=str, help="s: single, w: window")

    # Optimizer parameters
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--lr", type=float, default=1e-3, metavar="LR")
    # 模型保存地址
    parser.add_argument("--model_save_path", type=str, default="./saved_models/")
    return parser


def train(args, model, trn_dataset):
    # 创建trn_loader
    trn_loader = DataLoader(trn_dataset, batch_size=args.batch_size, shuffle=True)
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 训练模型
    model.to(device)
    for epoch in range(args.epochs):
        model.train()
        step = 0
        n_minibatch = math.ceil(len(trn_dataset) / args.batch_size)
        for i, (inputs, labels) in enumerate(trn_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            step += 1
            if step % args.detail_step == 0:
                print(
                    "epoch:[%d / %d] batch:[%d / %d] loss: %.3f lr: %.2e"
                    % (
                        epoch + 1,
                        args.epochs,
                        step,
                        n_minibatch,
                        loss,
                        optimizer.param_groups[0]["lr"],
                    )
                )
        # 保存模型
        if (epoch + 1) % 100 == 0:
            torch.save(
                model.state_dict(),
                args.model_save_path
                + args.model_name
                + "_model_{}_{}.pth".format(args.map_type_code, args.mode),
            )


def main(args):
    # 构建网络模型
    if args.map_type_code == "0x35":
        model = CustomNet(80, 60)
    elif args.map_type_code == "0x36":
        model = CustomNet(50, 60)

    # 读取数据
    trn_dataset = Dataset_load_trn(map_type_code=args.map_type_code, train=True)
    print("开始训练单时刻模型，图谱类型：{}".format(args.map_type_code))
    args.mode = "s"  # 单图谱
    # 训练模型
    train(args, model, trn_dataset)

    # 构建窗口数据集模型
    if args.map_type_code == "0x35":
        model = CustomWinNet(80, 60)
    elif args.map_type_code == "0x36":
        model = CustomWinNet(50, 60)

    # 读取窗口数据集
    trn_dataset = Dataset_load_window_trn(map_type_code=args.map_type_code, train=True)
    print("开始训练窗口模型，图谱类型：{}".format(args.map_type_code))
    args.mode = "w"  # 窗口
    # 训练窗口数据集模型
    train(args, model, trn_dataset)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    args.map_type_code = "0x35"
    main(args)
