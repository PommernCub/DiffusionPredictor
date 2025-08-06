# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import os
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# 配置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 超参数配置 (必须与训练时一致)
class Config:
    SEQ_LEN = 50          # 序列长度
    BATCH_SIZE = 64      # 批处理大小
    HIDDEN_SIZE = 64      # LSTM隐藏层大小
    NUM_LAYERS = 2        # LSTM层数
    DROPOUT = 0.2         # Dropout概率
    FEATURE_COLS = range(2, 13)  # 特征列索引 (3-13列)
    LABEL_COL = 13        # 标签列索引 (第14列)

config = Config()


# LSTM模型定义 (必须与训练时一致)
class DiffusionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(DiffusionLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 2)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        out = self.dropout(last_output)
        out = self.fc(out)
        return out


# 测试数据集类
class TestDataset(Dataset):
    def __init__(self, features):
        self.features = features
        self._create_sequences()
    
    def _create_sequences(self):
        num_samples = len(self.features) - config.SEQ_LEN + 1
        self.sequences = []
        
        for i in range(num_samples):
            seq = self.features[i:i+config.SEQ_LEN]
            self.sequences.append(seq)
        
        self.sequences = np.array(self.sequences)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx])


def calculate_classification_metrics(result_df):

    # 过滤掉无法预测的点（预测值为-1）
    valid_df = result_df[result_df['Prediction'] != -1].copy()    
    # 检查是否有有效数据
    if len(valid_df) == 0:
        print("警告：没有有效的预测结果（所有预测点均为-1）")
        return
    
    # 提取真实标签和预测结果
    true_labels = valid_df['True_Label'].values
    predictions = valid_df['Prediction'].values
    
    # 计算指标
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    cm = confusion_matrix(true_labels, predictions)  # 混淆矩阵
    
    # 打印结果
    print("\n分类指标:")
    print(f"有效预测点数: {len(valid_df)}")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall/Sensitivity): {recall:.4f}")
    print(f"F1 分数: {f1:.4f}")
    print("\n混淆矩阵:")
    print(f"        Predicted 0   Predicted 1")
    print(f"Actual 0: {cm[0, 0]:8d}      {cm[0, 1]:8d}")
    print(f"Actual 1: {cm[1, 0]:8d}      {cm[1, 1]:8d}")
    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Free (0)', 'Adsorbed (1)'], 
                yticklabels=['Free (0)', 'Adsorbed (1)'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'valid_count': len(valid_df)
    }


def main():
    
    start_time = time.time()
    # 加载测试数据
    test_file = 'test_data_point_cnn200_smooth.csv'
    print(f"Loading test data from {test_file}...")
    test_df = pd.read_csv(test_file)
    
    # 提取特征
    features = test_df.iloc[:, config.FEATURE_COLS].values
    
    # 加载标准化器并标准化特征
    print("Loading scaler...")
    scaler = joblib.load('scaler.pkl')
    features = scaler.transform(features)
    
    # 创建测试数据集
    test_dataset = TestDataset(features)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # 初始化模型
    print("Initializing model...")
    model = DiffusionLSTM(
        input_size=features.shape[1],
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT
    ).to(device)
    
    # 加载预训练权重
    print("Loading pretrained weights...")
    model.load_state_dict(torch.load('lstm_model.pth', map_location=device))
    model.eval()
    
    # 进行预测
    print("Running predictions...")
    all_preds = []
    
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
    
    # 8. 处理序列偏移
    all_preds = np.array(all_preds)
    padding = np.full(config.SEQ_LEN - 1, -1)  # 用-1填充无法预测的点
    full_preds = np.concatenate([padding, all_preds])
    
    # 保存预测结果
    result_df = pd.DataFrame({
        'Diffusion_Coef': test_df['Diffusion Coefficient'].values,  # 真实扩散系数
        'True_Label': test_df['is_normal'].values,  # 真实标签
        'Prediction': full_preds,      # 预测数值
        'Predicted_State': np.where(full_preds == 1, 'Adsorbed', 
                           np.where(full_preds == 0, 'Free', 'Not_Predicted'))  # 预测状态
    })
    
    # 保存完整结果
    output_file = 'predictions_LSTM_test.csv'
    result_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
    print(f"Total predictions: {len(full_preds)}")
    
    # 测试时间
    print(f"运行时间：{time.time() - start_time: .2f} seconds")    
    # 统计预测分布
    calculate_classification_metrics(result_df)
    pred_counts = pd.Series(full_preds).value_counts().sort_index()
    print("\nPrediction Distribution:")
    for val, count in pred_counts.items():
        state = "Adsorbed" if val == 1 else "Free" if val == 0 else "Not_Predicted"
        print(f"{state} ({val}): {count} points ({count/len(full_preds):.2%})")


if __name__ == "__main__":
    main()