import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score
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
    # 新结果文件编号为现有最大结果编号+1
    new_exp = f"exp{max_num + 1}"
    exp_path = os.path.join(base_dir, new_exp)
    os.makedirs(exp_path, exist_ok=True)
    return exp_path


def load_test_data(data_path, seq_length):
    """加载并预处理测试数据"""
    df = pd.read_csv(data_path, usecols=range(2, 14), header=0)
    raw_data = df.values.astype(np.float32)  # 将状态对应的布尔值转化为0/1    

    # 提取需要保留的原始数据段（特征和标签）, 处理方法要与训练时一致
    features = []
    labels = []
    step = seq_length - 4  # 每个窗口滑动步长
    total_windows = (len(raw_data) - seq_length) // step + 1
    for i in range(total_windows):
        segment = raw_data[i*step : i*step+seq_length]
        # 分别取出当前窗口片段的特征和标签, 作为1个样本：(该数据前2列为特征，最后一列为标签)
        features.append(segment[:, :-1])  # 取最后一列之前的特征列, 转换为(2, seq_len)
        labels.append(segment[:, -1].reshape(-1))      # 取最后一列中全部点的状态标签
    # 处理剩余数据（最后不足一个窗口的部分）
    remaining = len(raw_data) - ((total_windows-1) * step + seq_length)
    if remaining > 0:
        last_segment = raw_data[-seq_length:]  # 保持与前面每个样本长度一致
        features.append(last_segment[:, :-1])
        labels.append(last_segment[:, -1].reshape(-1))
    
    # 创建数据集
    test_dataset = dataset.DiffusionDataset(raw_data, seq_length)
    
    return test_dataset, features, labels

    
def main():
    # 配置参数
    SEQ_LENGTH = 50  # 需与训练时一致
    MODEL_PATH = "models/best_model.pth"
    TEST_DATA_PATH = "data/test_data_point_cnn.csv"
    RESULTS_FILENAME = "test-results.csv"
    NUM_FEATURES = 11  # 测试集文件特征数
    NUM_CLASSES = 2
    
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

    # 加载数据
    test_dataset, features, true_labels = load_test_data(TEST_DATA_PATH, SEQ_LENGTH)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 进行预测
    all_preds = []
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc='Batch')        
        for inputs, _ in progress_bar:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            flattened_preds = preds.cpu().numpy().flatten().tolist()
            
            features1 = np.concatenate([features[i].T[0] for i in range(len(features))])
            features2 = np.concatenate([features[i].T[1] for i in range(len(features))])            
            try:
                true_labels = np.concatenate(true_labels)
            except:
                pass
            all_preds.extend(flattened_preds)
    # 生成结果DataFrame
    results_df = pd.DataFrame({
        "feature1": features1,
        "feature2": features2,
        "true_label": true_labels,
        "predicted_label": all_preds
    })
    
    
    # 保存预测结果
    results_path = os.path.join(exp_path, RESULTS_FILENAME)
    results_df.to_csv(results_path, index=False)
    print(f"预测结果已保存至：{results_path}")

    # 计算评估指标
    accuracy = accuracy_score(true_labels, all_preds)
    precision = precision_score(true_labels, all_preds, average="binary")
    recall = recall_score(true_labels, all_preds, average="binary")
    f1 = f1_score(true_labels, all_preds, average="binary")

    # 打印并保存指标
    metrics = f"""模型性能评估：
                    Accuracy:  {accuracy:.4f}
                    Precision: {precision:.4f}
                    Recall:    {recall:.4f}
                    F1 Score:  {f1:.4f}"""
    
    metrics_path = os.path.join(exp_path, "Performances.txt")
    with open(metrics_path, "w") as f:
        f.write(metrics)
    
    print("\n" + metrics)
    print(f"\n评估指标已保存至：{metrics_path}")


if __name__ == "__main__":
    main()
