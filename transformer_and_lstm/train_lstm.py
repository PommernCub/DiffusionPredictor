# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt
import time

# 配置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 超参数配置
class Config:
    SEQ_LEN = 50          # 序列长度（与轨迹长度匹配）
    BATCH_SIZE = 64      # 批处理大小
    HIDDEN_SIZE = 64      # LSTM隐藏层大小
    NUM_LAYERS = 2        # LSTM层数
    NUM_EPOCHS = 10       # 训练轮数
    LR = 0.001            # 学习率
    DROPOUT = 0.2         # Dropout概率
    TEST_SIZE = 0.2       # 测试集比例
    FEATURE_COLS = range(2, 13)  # 特征列索引 (3-13列)
    LABEL_COL = 13        # 标签列索引 (第14列)

config = Config()

# 自定义数据集类（处理大文件）
class TrajectoryDataset(Dataset):
    def __init__(self, file_path, chunk_size=50000):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.scaler = StandardScaler()
        self._preprocess()
    
    def _preprocess(self):
        # 第一次遍历：拟合scaler
        print("Fitting scaler...")
        chunks = pd.read_csv(self.file_path, chunksize=self.chunk_size)
        for chunk in chunks:
            features = chunk.iloc[:, config.FEATURE_COLS].values
            self.scaler.partial_fit(features)
        
        # 第二次遍历：处理数据
        print("Processing data...")
        chunks = pd.read_csv(self.file_path, chunksize=self.chunk_size)
        self.all_features = []
        self.all_labels = []
        
        for chunk in chunks:
            # 提取特征和标签
            features = chunk.iloc[:, config.FEATURE_COLS].values
            labels = chunk.iloc[:, config.LABEL_COL].map({True: 1, False: 0}).values
            
            # 标准化特征
            features = self.scaler.transform(features)
            
            self.all_features.append(features)
            self.all_labels.append(labels)
        
        # 合并所有块
        self.features = np.concatenate(self.all_features)
        self.labels = np.concatenate(self.all_labels)
        
        # 转换为序列格式
        self._create_sequences()
    
    def _create_sequences(self):
        num_samples = len(self.features) - config.SEQ_LEN + 1
        self.sequences = []
        self.seq_labels = []
        
        for i in range(num_samples):
            seq = self.features[i:i+config.SEQ_LEN]
            label = self.labels[i+config.SEQ_LEN-1]  # 序列最后一个点的标签
            self.sequences.append(seq)
            self.seq_labels.append(label)
        
        self.sequences = np.array(self.sequences)
        self.seq_labels = np.array(self.seq_labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.LongTensor([self.seq_labels[idx]])
        )


# LSTM模型定义
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
        # x形状: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)  # lstm_out形状: (batch_size, seq_len, hidden_size)
        
        # 只取序列最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        
        out = self.dropout(last_output)
        out = self.fc(out)  # 形状: (batch_size, 2)
        return out


# 训练函数
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.squeeze(1).to(device)  # 移除多余的维度
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计信息
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')


# 评估函数
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.squeeze(1).to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels)


# 主函数
def main():
    
    start_time = time.time()
    # 1. 加载和预处理数据
    dataset = TrajectoryDataset('train_data_point_cnn200_smooth.csv')
    
    # 2. 分割数据集
    indices = list(range(len(dataset)))
    train_idx, test_idx = train_test_split(indices, test_size=config.TEST_SIZE, shuffle=False)
    
    train_set = torch.utils.data.Subset(dataset, train_idx)
    test_set = torch.utils.data.Subset(dataset, test_idx)
    
    # 3. 创建数据加载器
    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # 4. 初始化模型
    input_size = dataset.features.shape[1]  # 特征数量
    model = DiffusionLSTM(
        input_size=input_size,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT
    ).to(device)
    
    # 5. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    
    # 6. 训练模型
    train_model(model, train_loader, criterion, optimizer, config.NUM_EPOCHS)
    
    # 7. 评估模型
    test_preds, test_labels = evaluate_model(model, test_loader)
    
    # 8. 输出全数据集的预测结果
    full_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    full_preds, _ = evaluate_model(model, full_loader)
    
    # 处理序列偏移：前SEQ_LEN-1个点没有预测
    padding = np.full(config.SEQ_LEN - 1, -1)  # 用-1填充无法预测的点
    full_preds = np.concatenate([padding, full_preds])
    
    # 保存预测结果
    output_df = pd.DataFrame({'Prediction': full_preds})
    output_df.to_csv('predictions_LSTM.csv', index=False)
    print("Predictions saved to predictions.csv")
    # 保存模型文件
    torch.save(model.state_dict(), 'lstm_model.pth')
    joblib.dump(dataset.scaler, 'scaler.pkl')
    
    # 训练时间
    print(f"运行时间：{time.time() - start_time: .2f} seconds")
    
    # 9. 可视化部分结果
    plt.figure(figsize=(15, 6))
    plt.plot(dataset.labels[50:150], 'b-', label='True Labels')
    plt.plot(full_preds[50:150], 'r--', alpha=0.7, label='Predictions')
    plt.title('Diffusion State Prediction')
    plt.ylabel('State (0=Free, 1=Adsorbed)')
    plt.xlabel('Time Step')
    plt.legend()
    plt.savefig('prediction_plot.png')
    plt.show()

if __name__ == "__main__":
    main()