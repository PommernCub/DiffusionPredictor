import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
import time
import csv

# 从utils包导入模块
from utils import modules, dataset, trainer


def load_data(config):
    # 数据加载逻辑
    ABS_PATH = os.getcwd()
    train_df = pd.read_csv(
        os.path.join(ABS_PATH, 'data', 'train_data_point_cnn.csv'), 
        usecols=range(2, 16),  # 读取输入特征+标签列
    )
    test_df = pd.read_csv(
        os.path.join(ABS_PATH, 'data', 'test_data_point_cnn.csv'), 
        usecols=range(2, 16),  # 读取输入特征+标签列
    )
    return (
        # 转换为float32, 优化计算; 同时将标签布尔值转化为0/1
        train_df.values.astype(np.float32),
        test_df.values.astype(np.float32)
    )


def main():
    # 配置参数
    config = {
        "seed": 42,
        "seq_length": 50,  # 单个样本长度
        "batch_size": 64,
        "epochs": 100,
        "lr": 1e-3,
        "valid_ratio": 0.2,
        "num_workers": 4,
        "num_features": 13,  # 输入训练数据文件的特征数
        "num_classes": 2,
        "log_file": "training_log.csv",  # 生成loss函数文件
        "model_dir": "models",  # 模型存储文件夹
    }

    if not os.path.exists(config["model_dir"]):
        os.mkdir(config["model_dir"])
    # 初始化设置
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    device = trainer.setup_device()
    with open(config["log_file"], "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train_Loss", "Val_Loss", "Val_Acc"])

    # 加载数据
    train_data, test_data = load_data(config)
    train_dataset = dataset.DiffusionDataset(train_data, config["seq_length"])
    valid_dataset = dataset.DiffusionDataset(test_data, config["seq_length"])


    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"]
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"]
    )

    # 初始化模型
    model = modules.DiffusionPredictor(
            input_channels=config["num_features"], 
            num_classes=config["num_classes"]
            ).to(device)
    print(f"Model architecture:\n{model}")  # 打印模型结构
    criterion = torch.nn.CrossEntropyLoss()
    optimizer, scheduler = trainer.create_optimizer(model, config["lr"])

    time_start = time.time()
    print(f"Start Training at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}...")
    # 训练循环
    best_val_loss = float("inf")
    patience, no_improve = 10, 0  # 添加用于早停的patience机制
    no_improve = 0
    for epoch in range(config["epochs"]):
            
        train_loss = trainer.train_epoch(
            (epoch, config["epochs"]), model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = trainer.validate(
            (epoch, config["epochs"]), model, valid_loader, criterion, device)
        
        # 将结果写入CSV文件
        with open(config["log_file"], "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, 
                f"{train_loss:.4f}", 
                f"{val_loss:.4f}", 
                f"{val_acc:.2f}%"
            ])
        
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/best_model.pth")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping!")
                break
        
        print(f"Epoch {epoch+1}/{config['epochs']}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print("-" * 50)
    # 结束训练打印用时
    print(f"Finished Training at {time.time() - time_start: .2f} seconds ")


if __name__ == "__main__":
    main()
