import torch
import torch.nn as nn
import torch.nn.functional as F



class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.LayerNorm(in_channels // reduction),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )
        self.conv = nn.Conv1d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        b, c, _ = x.size()
        avg = self.gap(x).view(b, c)
        max_ = self.gmp(x).view(b, c)
        y = self.fc(avg + max_).view(b, c, 1)
        
        # 空间注意力
        spatial = torch.cat([x.mean(dim=1, keepdim=True), x.max(dim=1, keepdim=True)[0]], dim=1)
        spatial = self.conv(spatial).sigmoid()
        
        return x * y * spatial  # 联合通道和空间注意力
    

class TemporalLSTM(nn.Module):
    def __init__(self, in_channels, hidden_size=128, bidirectional=True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.attn = nn.Sequential(
            nn.Linear(2*hidden_size if bidirectional else hidden_size, 1),
            nn.Softmax(dim=1)
        )
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        # x shape: (B, C, T)
        x = x.permute(0, 2, 1)  # (B, T, C)
        out, _ = self.lstm(x)
        out = self.drop(out)
        return out.permute(0, 2, 1)  # (B, C, T)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        padding = (kernel_size + (kernel_size-1)*(dilation-1)) // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, 
                     padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            # nn.ReLU(inplace=True)
            nn.LeakyReLU(negative_slope=0.1, inplace=False)  # ReLu 输入负数可能导致梯度消失
        )
    
    def forward(self, x):
        return self.conv(x)
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel_size, 1)        
        # 添加通道维度的跳跃连接
        self.channel_att = ChannelAttention(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )
        # 时间维度的跳跃连接
        self.temporal_conv = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        # 正则化
        self.drop = nn.Dropout(0.2)
    
    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.channel_att(x)
        x = x + F.leaky_relu(self.temporal_conv(residual), 0.1)
        # return F.leaky_relu(x + residual, negative_slope=0.1)
        return self.drop(x)



class MultiScaleFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branches = nn.ModuleList([
            ConvBlock(in_channels, in_channels//4, 3, dilation=1),
            ConvBlock(in_channels, in_channels//4, 5, dilation=1),
            ConvBlock(in_channels, in_channels//4, 3, dilation=2),
            ConvBlock(in_channels, in_channels//4, 1)])
        
        self.fusion = nn.Sequential(
            ChannelAttention(in_channels),
            nn.Conv1d(in_channels, in_channels, 1))
    
    def forward(self, x):
        features = [branch(x) for branch in self.branches]
        fused = torch.cat(features, dim=1)
        return self.fusion(fused)
    

class DiffusionPredictor(nn.Module):
    def __init__(self, input_channels=2, num_classes=2):
        super().__init__()
        """
        - 开始输入特征矩阵维度: (Batch, C, T), 注意C是特征数,使用 input_channels 定义
          对应数据的特征数 (步长, 平均扩散系数), 如改变特征数这里须修改;
        - Pytorch 1DConv 默认在最后一个维度上卷积, 逐通道进行, 通过out_channels指定卷积后的通道数; 
          即 1DConv 仅改变第二个维度 num_Features 的通道数
        """
        # Stage 1: Shallow features
        self.stage1 = nn.Sequential(
            ConvBlock(input_channels, 64, 1), # (Batch, 64, T), k=1
            ConvBlock(64, 64, 3), # (Batch, 64, T), k=3
            nn.MaxPool1d(2, ceil_mode=True)  # (Batch, 64, T/2) ceil_mode保持时间维度长度
        )
        
        # Stage 2: Mid-level features
        self.stage2 = nn.Sequential(
            ResidualBlock(64, 128, kernel_size=5),
            TemporalLSTM(128, hidden_size=64),  # LSTM特征提取
            nn.MaxPool1d(2, ceil_mode=True)
        )
        
        # Stage 3: Deep features
        self.stage3 = ResidualBlock(128, 256, kernel_size=3, dilation=2)
        # Multi-scale Fusion
        self.fusion = MultiScaleFusion(256)
        
        # Classification Head
        self.head = nn.Sequential(
            ConvBlock(256, 128, 3),
            nn.LeakyReLU(0.1),
            # nn.LayerNorm(128),
            nn.Dropout(0.3),
            nn.Conv1d(128, num_classes, 1)
        )
    
    def forward(self, x):
        # input x shape: (B, N, T)
        Seq_length = x.size(-1)
        x = self.stage1(x)  # -> (B, 64, T/2)        
        x = self.stage2(x)  # -> (B, 128, T/4)
        x = self.stage3(x)  # -> (B, 256, T/4)     
        
        x = F.interpolate(x, size=Seq_length, mode='linear', align_corners=True)  # 恢复原始时间维度
        x = self.fusion(x)  # 多尺度融合
        return self.head(x)  # (B, C, T)
