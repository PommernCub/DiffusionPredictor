# -*- coding: utf-8 -*-
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 配置参数
class Config:
    # 数据参数
    feature_cols = range(2, 13)  # 第3-13列作为特征 (0-based索引)
    label_col = 13               # 第14列作为标签 (0-based索引)
    window_size = 50             # 轨迹窗口大小
    
    # 模型参数
    input_dim = 11               # 输入特征维度
    d_model = 128                # Transformer特征维度
    nhead = 8                    # 注意力头数
    num_layers = 6               # Transformer层数
    dim_feedforward = 512        # 前馈网络维度
    dropout = 0.2                # Dropout概率
    model_save_path = 'best_transformer_model.pth'
    
    # 训练参数
    batch_size = 64
    lr = 0.001
    epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 自定义数据集类
class TrajectoryDataset(Dataset):
    def __init__(self, features, labels, window_size):
        self.features = features
        self.labels = labels
        self.window_size = window_size
        
    def __len__(self):
        return len(self.features) - self.window_size + 1
    
    def __getitem__(self, idx):
        # 提取滑动窗口内的特征和标签
        x = self.features[idx:idx+self.window_size]
        y = self.labels[idx:idx+self.window_size]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        return x + self.pe[:x.size(1)].permute(1, 0, 2)


# Transformer分类模型
class DiffusionClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 输入嵌入层
        self.embedding = nn.Linear(config.input_dim, config.d_model)
        # 位置编码
        self.pos_encoder = PositionalEncoding(config.d_model, config.window_size)
        
        # Transformer编码器
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
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x形状: [batch_size, window_size, input_dim]
        x = self.embedding(x)  # 转换为 [batch_size, window_size, d_model]
        x = self.pos_encoder(x)  # 位置编码
        x = self.transformer(x)
        
        # 逐点分类
        # 使用切片保留第二维（时间步维度）
        return self.classifier(x).squeeze(-1)  # 输出形状: [batch_size, window_size]


# 训练流程
def main():
    
    time_start = time.time()
    config = Config()
    
    # 数据加载与预处理
    print("Loading data...")
    df = pd.read_csv('train_data_point_cnn200_smooth.csv')
    # df = pd.read_csv('../250521_mixx/data/train_data_point_cnn40_smooth.csv')
    
    # 提取特征和标签
    features = df.iloc[:, config.feature_cols].values
    labels = df.iloc[:, config.label_col].values
    
    # 转换标签为数值 (取反)
    try:  # labels 读取是bool型
        labels = (~labels).astype(np.float32)
    except:  # labels 读取是字符串
        labels = (np.where(labels == 'True', 0.0, 1.0)).astype(np.float32)
    
    # 标准化特征
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    # 保存标准化器，用于测试
    np.save('feature_scaler.npy', scaler.mean_)
    np.save('feature_std.npy', scaler.scale_)
    
    # 创建数据集
    print("Creating datasets...")
    dataset = TrajectoryDataset(features, labels, config.window_size)
    
    # 划分训练集和验证集 (按时间顺序划分)
    # train_size = int(0.8 * len(dataset))
    # val_size = len(dataset) - train_size
    # train_dataset, val_dataset = torch.utils.data.random_split(
    #     dataset, [train_size, val_size], 
    #     generator=torch.Generator().manual_seed(42)
    # )
    train_ratio = 0.8
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)    
    # 直接使用切片创建训练集和验证集
    train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, total_size))
    
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    
    # 初始化模型
    model = DiffusionClassifier(config).to(config.device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    
    best_val_acc = 0.0    
    print("Starting training...")
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(config.device), batch_y.to(config.device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
        
        # validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(config.device), batch_y.to(config.device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                
                # 计算准确率
                predicted = (outputs > 0.5).float()
                correct += (predicted == batch_y).sum().item()
                total += batch_y.numel()
        
        # 打印统计信息
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        val_acc = correct / total
        
        print(f'Epoch {epoch+1}/{config.epochs}: '
              f'Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'Accuracy: {val_acc:.4f}')
        
        # 保存最佳模型文件
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'config': config,
            }, config.model_save_path)
            print(f"Saved best model with val acc: {val_acc:.4f}")
    
    print(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")
    
    # 全数据集预测
    print("Generating full predictions...")
    model.eval()
    all_predictions = np.zeros(len(features))
    
    # 使用滑动窗口进行预测
    with torch.no_grad():
        # 处理完整数据
        full_dataset = TrajectoryDataset(features, labels, config.window_size)
        full_loader = DataLoader(full_dataset, batch_size=1024, shuffle=False)
        
        start_idx = 0
        for batch_x, _ in full_loader:
            batch_x = batch_x.to(config.device)
            outputs = model(batch_x).cpu().numpy()
            
            # 对于每个窗口，取中间点的预测结果
            mid_point = config.window_size // 2
            batch_size = outputs.shape[0]
            
            # 填充预测结果
            for i in range(batch_size):
                idx = start_idx + i + mid_point
                if idx < len(all_predictions):
                    all_predictions[idx] = outputs[i, mid_point]
            
            start_idx += batch_size
    
    # 处理边界点 (使用最近的预测)
    all_predictions[:config.window_size//2] = all_predictions[config.window_size//2]
    all_predictions[-config.window_size//2:] = all_predictions[-config.window_size//2-1]
    
    # 保存结果
    result_df = pd.DataFrame({
        'Df': df['Diffusion Coefficient'].values,
        'true_label': labels,
        'prediction': (all_predictions > 0.5).astype(int),
        'probability': np.round(all_predictions, 4)
    })
    result_df.to_csv('predictions_transformer.csv', index=False)
    print("Predictions saved to predictions.csv")
    print(f"运行时间：{time.time()-time_start: .2f}")

if __name__ == "__main__":
    main()
