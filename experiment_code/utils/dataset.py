# -*- coding: utf-8 -*-
"""
Created on Wed May 21 16:26:24 2025

@author: 91278
"""
import torch
from torch.utils.data import Dataset
import numpy as np


class DiffusionDataset(Dataset):
    def __init__(self, data, seq_length=100):
        """
        data: 原始数据，形状为(N, 4)
        seq_length: 分割后的序列长度
        """
        self.features = []
        self.labels = []
      
        # 数据预处理：使用带重叠的滑动窗口分割序列
        step = seq_length - 4  # 每个窗口滑动步长
        total_windows = (len(data) - seq_length) // step + 1
        for i in range(total_windows):
            segment = data[i*step : i*step+seq_length]
            # 分别取出当前窗口片段的特征和标签, 作为1个样本：(该数据前N列为特征，最后一列为标签)
            self.features.append(segment[:, :-1].T)  # 取最后一列之前的特征列, 转换为(N, seq_len)
            self.labels.append(segment[:, -1])      # 取最后一列中全部点的状态标签
        # 处理剩余数据（最后不足一个窗口的部分）
        remaining = len(data) - ((total_windows-1) * step + seq_length)
        if remaining > 0:
            last_segment = data[-seq_length:]  # 保持与前面每个样本长度一致
            self.features.append(last_segment[:, :-1].T)
            self.labels.append(last_segment[:, -1])
        
        # 将列表转为Numpy数组, 再转为Tensor(一次性转换)
        self.features = np.stack(self.features)  # 形状 (num_samples, 2, seq_length)
        self.features = torch.FloatTensor(self.features)
        self.labels = np.stack(self.labels)      # 形状 (num_samples, seq_length)        
        self.labels = torch.LongTensor(self.labels)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
