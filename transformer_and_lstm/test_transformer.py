# -*- coding: utf-8 -*-
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# 使用与训练相同的配置类
class Config:
    feature_cols = range(2, 13)
    label_col = 13
    window_size = 50
    input_dim = 11
    d_model = 128
    nhead = 8
    num_layers = 6
    dim_feedforward = 512
    dropout = 0.1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_save_path = 'best_transformer_model.pth'


# 使用与训练相同的模型类
class DiffusionClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Linear(config.input_dim, config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.classifier(x).squeeze(-1)


# 使用与训练相同的数据集类
class TrajectoryDataset(Dataset):
    def __init__(self, features, window_size):
        self.features = features
        self.window_size = window_size
        
    def __len__(self):
        return len(self.features) - self.window_size + 1
    
    def __getitem__(self, idx):
        x = self.features[idx:idx+self.window_size]
        return torch.tensor(x, dtype=torch.float32)


def calculate_classification_metrics(result_df):

    # 提取真实标签和预测结果
    true_labels = result_df['True_Label'].values
    predictions = result_df['Prediction'].values
    
    # 计算指标
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    cm = confusion_matrix(true_labels, predictions)  # 混淆矩阵
    
    # 打印结果
    print("\n分类指标:")
    print(f"预测点数: {len(result_df)}")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall/Sensitivity): {recall:.4f}")
    print(f"F1 分数: {f1:.4f}")
    print("\n混淆矩阵:")
    print(f"        Predicted 0   Predicted 1")
    print(f"Actual 0: {cm[0, 0]:8d}      {cm[0, 1]:8d}")
    print(f"Actual 1: {cm[1, 0]:8d}      {cm[1, 1]:8d}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'valid_count': len(result_df)
    }


def main():
    config = Config()
    
    start_time = time.time()
    # 1. 加载测试数据
    print("Loading test data...")
    test_df = pd.read_csv('test_data_point_cnn200_smooth.csv')  # 替换为测试数据路径
    
    # 提取特征和原始标签
    features = test_df.iloc[:, config.feature_cols].values
    raw_labels = test_df.iloc[:, config.label_col].values  # 原始标签字符串
    
    # 2. 预处理
    print("Preprocessing data...")
    # 加载标准化器
    scaler_mean = np.load('feature_scaler.npy')
    scaler_std = np.load('feature_std.npy')
    features = (features - scaler_mean) / scaler_std
    
    # 3. 加载模型
    print("Loading model...")
    model = DiffusionClassifier(config).to(config.device)
    checkpoint = torch.load(config.model_save_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 4. 创建测试数据集
    print("Creating test dataset...")
    test_dataset = TrajectoryDataset(features, config.window_size)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    
    # 5. 进行预测
    print("Generating predictions...")
    all_predictions = np.zeros(len(features))
    all_probabilities = np.zeros(len(features))
    
    with torch.no_grad():
        start_idx = 0
        for batch_x in test_loader:
            batch_x = batch_x.to(config.device)
            outputs = model(batch_x).cpu().numpy()
            
            # 对于每个窗口，取中间点的预测结果
            mid_point = config.window_size // 2
            batch_size = outputs.shape[0]
            
            # 填充预测结果和概率
            for i in range(batch_size):
                idx = start_idx + i + mid_point
                if idx < len(all_predictions):
                    all_predictions[idx] = outputs[i, mid_point] > 0.5
                    all_probabilities[idx] = outputs[i, mid_point]
            
            start_idx += batch_size
    
    # 处理边界点
    first_valid = config.window_size // 2
    all_predictions[:first_valid] = all_predictions[first_valid]
    all_probabilities[:first_valid] = all_probabilities[first_valid]
    
    last_valid = len(features) - config.window_size // 2 - 1
    all_predictions[last_valid+1:] = all_predictions[last_valid]
    all_probabilities[last_valid+1:] = all_probabilities[last_valid]
    
    # 6. 保存结果
    print("Saving results...")
    # 反转标签含义
    # 模型预测: 1 = 吸附 (对应原始标签中的False)
    #           0 = 自由扩散 (对应原始标签中的true)
    
    # 创建结果DataFrame
    result_df = pd.DataFrame({
        'Diffusion_Coef': test_df['Diffusion Coefficient'].values,  # 原始扩散系数
        'True_Label': ~raw_labels,  # 原始标签(取反)
        'Prediction': all_predictions,  # 预测标签
        'Probability': np.round(all_probabilities, 4)  # 吸附状态概率
    })
    
    # 保存到CSV
    result_df.to_csv('predictions_transformer_test.csv', index=False)
    print("Predictions saved to predictions_transformer.csv")
    print(f"运行时长：{time.time() - start_time: .2f} seconds")    
    # 统计预测分布
    calculate_classification_metrics(result_df)

if __name__ == "__main__":
    main()
