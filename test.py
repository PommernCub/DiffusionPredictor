import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, r2_score
)
from utils import modules, dataset

def create_exp_dir(base_dir="results"):
    """自动创建实验目录"""
    os.makedirs(base_dir, exist_ok=True)
    
    # 获取现有实验编号
    existing_exps = [d for d in os.listdir(base_dir) 
                    if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("exp")]
    
    max_num = 0
    for exp in existing_exps:
        try:
            num = int(exp[3:])
            max_num = max(max_num, num)
        except ValueError:
            continue
    
    new_exp = f"exp{max_num + 1}"
    exp_path = os.path.join(base_dir, new_exp)
    os.makedirs(exp_path, exist_ok=True)
    return exp_path

def load_test_data(data_path, seq_length):
    """加载并预处理测试数据"""
    df = pd.read_csv(data_path, usecols=range(2, 14), header=0)
    raw_data = df.values.astype(np.float32)
    
    # 创建数据集
    test_dataset = dataset.DiffusionDataset(raw_data, seq_length)
    
    # 提取特征和标签用于结果保存
    features = []
    labels = []
    step = seq_length - 4
    total_windows = (len(raw_data) - seq_length) // step + 1
    
    for i in range(total_windows):
        segment = raw_data[i*step : i*step+seq_length]
        features.append(segment[:, :-1])
        labels.append(segment[:, -1].reshape(-1))
    
    # 处理剩余数据
    remaining = len(raw_data) - ((total_windows-1) * step + seq_length)
    if remaining > 0:
        last_segment = raw_data[-seq_length:]
        features.append(last_segment[:, :-1])
        labels.append(last_segment[:, -1].reshape(-1))
    
    return test_dataset, features, labels

def test_model_for_size(model, device, particle_size, exp_path, seq_length=50):
    """测试特定粒子尺寸的数据集"""
    # 构建数据路径
    test_data_path = f"data/test_data_point_cnn{particle_size}_smooth.csv"
    print(f"\n正在测试粒子尺寸: {particle_size} | 数据路径: {test_data_path}")
    
    # 加载数据
    test_dataset, features, true_labels = load_test_data(test_data_path, seq_length)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 进行预测
    all_preds = []
    with torch.no_grad():
        for inputs, _ in tqdm(test_loader, desc=f'测试尺寸 {particle_size}'):
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy().flatten().tolist())
    
    # 准备结果数据
    features1 = np.concatenate([f.T[0] for f in features])
    features2 = np.concatenate([f.T[1] for f in features])
    true_labels_flat = np.concatenate(true_labels)
    
    # 生成结果DataFrame
    results_df = pd.DataFrame({
        "feature1": features1,
        "feature2": features2,
        "true_label": true_labels_flat,
        "predicted_label": all_preds
    })
    
    # 保存预测结果
    results_path = os.path.join(exp_path, f"results_{particle_size}.csv")
    results_df.to_csv(results_path, index=False)
    print(f"预测结果已保存至：{results_path}")
    
    # 计算评估指标
    accuracy = accuracy_score(true_labels_flat, all_preds)
    precision = precision_score(true_labels_flat, all_preds, average="binary")
    recall = recall_score(true_labels_flat, all_preds, average="binary")
    f1 = f1_score(true_labels_flat, all_preds, average="binary")
    r2 = r2_score(true_labels_flat, all_preds)
    
    # 返回指标和粒子尺寸
    return {
        "particle_size": particle_size,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "r2": r2
    }

def main():
    # 配置参数
    SEQ_LENGTH = 50
    MODEL_PATH = "best_model.pth"
    NUM_FEATURES = 11
    NUM_CLASSES = 2
    PARTICLE_SIZES = [24, 30, 40, 50, 100, 200, 500]  # 所有需要测试的粒子尺寸
    
    # 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建实验目录
    exp_path = create_exp_dir()
    print(f"实验目录：{exp_path}")
    
    # 加载模型
    model = modules.DiffusionPredictor(
        input_channels=NUM_FEATURES, 
        num_classes=NUM_CLASSES
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    # 存储所有结果的指标
    all_metrics = []
    
    # 对每个粒子尺寸进行测试
    for size in PARTICLE_SIZES:
        metrics = test_model_for_size(model, device, size, exp_path, SEQ_LENGTH)
        all_metrics.append(metrics)
    
    # 保存所有指标到文件
    metrics_path = os.path.join(exp_path, "Performances.txt")
    with open(metrics_path, "w") as f:
        f.write("Particle Size | Accuracy | Precision | Recall | F1 Score | R2 Score\n")
        f.write("-------------------------------------------------------------\n")
        
        for metrics in all_metrics:
            line = (
                f"{metrics['particle_size']:>10} | "
                f"{metrics['accuracy']:.4f} | "
                f"{metrics['precision']:.4f} | "
                f"{metrics['recall']:.4f} | "
                f"{metrics['f1']:.4f} | "
                f"{metrics['r2']:.4f}\n"
            )
            f.write(line)
    
    # 打印汇总结果
    print("\n所有粒子尺寸测试完成！")
    print("=" * 70)
    print("粒子尺寸 | Accuracy | Precision | Recall | F1 Score | R2 Score")
    print("-" * 70)
    for metrics in all_metrics:
        print(
            f"{metrics['particle_size']:>10} | "
            f"{metrics['accuracy']:.4f} | "
            f"{metrics['precision']:.4f} | "
            f"{metrics['recall']:.4f} | "
            f"{metrics['f1']:.4f} | "
            f"{metrics['r2']:.4f}"
        )
    
    print(f"\n完整指标已保存至：{metrics_path}")

if __name__ == "__main__":
    main()