# -*- coding: utf-8 -*-
"""
Created on Fri May 23 19:25:05 2025

@author: 91278
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import PowerTransformer
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d


def smooth_data(data, method='savgol', window_size=11, sigma=2.0, polyorder=2):
    """
   对二维轨迹数据进行平滑处理以消除噪声
   
   参数:
       trajectory (np.ndarray): 输入轨迹数据，形状为(N, 2)
       method (str): 平滑方法，可选'savgol'(推荐)/'gaussian'/'moving_average'
       window_size (int): 滑动窗口大小（建议奇数）
       sigma (float): 高斯滤波标准差（仅对gaussian有效）
       polyorder (int): 多项式阶数（仅对savgol有效）
       
   返回:
       np.ndarray: 平滑后的轨迹数据，形状与输入相同
    """
    data = np.asarray(data).flatten()
    n_points = len(data)
    if method == 'savgol':
        # 自动调整窗口为奇数
        window_size = window_size + 1 if window_size % 2 == 0 else window_size
        window_size = min(max(window_size, 3), n_points)
        return savgol_filter(data, window_size, polyorder)
    elif method == 'gaussian':
        return gaussian_filter1d(data, sigma=sigma)
    elif method == 'moving_average':
        window = np.ones(window_size)/window_size
        pad_width = window_size//2
        padded = np.pad(data, pad_width, mode='edge')
        smoothed = np.convolve(padded, window, mode='valid')
        return smoothed[:n_points]  # 修正边界截断
    else:
        raise ValueError("未知方法，可选：'savgol', 'gaussian', 'moving_average'")
    return smoothed

    


def calc_Euclid_distances(data):
    # 计算当前轨迹文件的所有步长 displacements
    dx = np.array(data['X'][1:]) - np.array(data['X'][:-1])
    dy = np.array(data['Y'][1:]) - np.array(data['Y'].iloc[:-1])
    dist = np.sqrt(dx**2 + dy**2)  # 欧氏距离
    # 使用计算的第二个欧氏距离填充第一个值
    dist = np.insert(dist, 0, dist[0] if len(dist) > 0 else 0)

    # 标签分类（支持浮点标签）
    labels = data['Diffusion Coefficient']
    is_normal = labels >= 0.9999  # 扩散系数标签为1的数据
    
    return dist, dist[is_normal], dist[~is_normal]


def calc_mean_displacement(data, window):
    """计算滑动窗口平均步长"""   
    mean_dist = np.array([], dtype=np.float32)
    std_dist = np.array([], dtype=np.float32)
    for i in range(0, len(data)):
        # 计算i点附近滑动窗口内的平均欧式距离
        if i >= window//2 and i <= len(data)-window//2-1:
            start = i - window // 2
            end = i + window // 2 + 1    
        else:
            start = max(i - window // 2, 0)
            end = min(len(data), i + window // 2 + 1)
        mean = data[start:end].mean().astype('float32')
        stderr = data[start:end].std().astype('float32')
        mean_dist = np.append(mean_dist, mean)
        std_dist = np.append(std_dist, stderr)
    
    # return mean_dist, std_dist
    return smooth_data(mean_dist), smooth_data(std_dist)


def normalize_data(X, normal_mean, normal_std,
    total_min, total_max, mode='std_norm'):
    """数据(标准)归一化"""
    if mode == 'custom':
        X_norm = (X - normal_mean) / normal_std        
        X_norm = (X_norm - total_min) / (total_max - total_min + 1e-6)
    elif mode == 'std':
        X_norm = (X - np.mean(X)) / np.std(X)    
    elif mode == 'norm':
        X_range = np.max(X) - np.min(X) + 1e-6  # 防止除0
        X_norm = (X - np.min(X)) / X_range        
    else: # mode == 'std_norm':
        X_norm = (X - np.mean(X)) / np.std(X)
        X_range = np.max(X_norm) - np.min(X_norm) + 1e-6
        X_norm = (X_norm - np.min(X_norm)) / X_range
        
    return X_norm


def power_transform(X):
    """对(步长)数据进行幂变换，增强数据分布的正态性"""
    X = np.array(X)
    pt = PowerTransformer()
    X_trans = pt.fit_transform(X.reshape(-1, 1)).flatten()  # 输出长度不变
    # return X_trans
    return smooth_data(X_trans)


def calc_gaussian_kde(X_label0, X_label1):
    """计算训练集两种分布的高斯核函数, 生成0/1两类标签的概率分布"""
    n0, n1 = len(X_label0), len(X_label1)
    std0, std1 = np.std(X_label0), np.std(X_label1)
    # 计算算最佳带宽（Silverman规则）
    bw0 = 1.06*std0*n0**(-1/5)
    bw1 = 1.06*std1*n1**(-1/5)
    kde_x0 = gaussian_kde(X_label0, bw_method=bw0)
    kde_x1 = gaussian_kde(X_label1, bw_method=bw1)
    return kde_x0, kde_x1


def calc_kde_features(kde_x0, kde_x1, X):
    """分辨该点属于0/1类别的概率密度"""
    X_kde0 = kde_x0.evaluate(X)  # X属于Class 0的概率密度 
    X_kde1 = kde_x1.evaluate(X)  # X属于Class 1的概率密度
    ratio = X_kde1 / (X_kde0 + 1e-6)  # 防止除零
    return X_kde0, X_kde1


def calc_first_differences(X):
    """计算步长数据的一阶差分"""
    X_diff = np.diff(X, n=1) 
    # 一阶差分第一个数据为0, 用第二个代替
    X_diff[0] = X_diff[1]
    # 缺失数据用计算出的最后一个数据填充
    X_diff_pad = np.append(X_diff, X_diff[-1])
    
    # return X_diff_pad
    return smooth_data(X_diff_pad)


def calc_fft_features(X, sliding_window):
    """通过傅里叶变换计算幅频分布特征"""
    # 全局傅里叶变换
    # fft_X = np.fft.fft(X)  # 包含振幅和相位信息的复数数组, 后N/2个点是前N/2的共轭镜像
    # amp_X = np.abs(fft_X)  # 全长幅频
    # angle_X = np.angle(fft_X)  # 相位角
    
    # 滑动窗口局部频域特征
    amp_features = []
    for i in range(len(X)):
        if i < sliding_window//2:
            window = X[:sliding_window]
        elif i > len(X)-sliding_window//2-1:
            window = X[len(X)-sliding_window:len(X)]
        else:
            window = X[i-sliding_window//2:i+sliding_window//2+1]        
        amp = np.abs(np.fft.fft(window))
        # 有效特征频点为窗口长度的前一半 (sliding_window//2+1)
        amp_features.append(amp[:sliding_window//2+1])
    # sliding_window=5 时, 有效频点为前3个点 (直流分量, 基频, 二次谐波)
    amp = np.array(amp_features).T
    
    for i in range(sliding_window//2+1):
        amp[i] = smooth_data(amp[i])

    return amp


def calc_autocorrelation(X, sliding_window, lag=1):
    """计算窗口内步长的自相关系数"""
    auto_cors = []
    if sliding_window <= lag:
        print(f"Error: 计算自相关函数, 窗口长度太短无法计算: \
              窗口长度应大于 {lag}, 当前窗口长度 {sliding_window}")
        return 0  # 窗口过短无法计算
    for i in range(len(X)):
        if i < sliding_window//2:
            window = X[:sliding_window]
        elif i > len(X)-sliding_window//2-1:
            window = X[len(X)-sliding_window:len(X)]
        else:
            window = X[i-sliding_window//2:i+sliding_window//2+1]
        mean = np.mean(window)
        numerator = np.sum((window[:sliding_window-lag] - mean) * (window[lag:] - mean))
        denominator = np.sum((window - mean)**2)
        auto_cors.append(numerator / denominator if denominator != 0 else 0)
    auto_cor_matrix = np.array(auto_cors, dtype='float32')
    
    # return auto_cor_matrix
    return smooth_data(auto_cor_matrix)


def plot_process_data(data, label):
    t = np.linspace(0, len(data)-1, len(data))
    clean = np.where(label > 0.9999, 0.3162, 0.2582)
    noisy = data
    
    # 应用不同平滑方法
    savgol_smoothed = smooth_data(noisy, method='savgol', window_size=11)
    gaussian_smoothed = smooth_data(noisy, method='gaussian', sigma=2)
    ma_smoothed = smooth_data(noisy, method='moving_average', window_size=7)
    
    # 可视化结果
    plt.figure(figsize=(10, 6))
    plt.plot(t, noisy, 'gray', lw=2, alpha=0.3, label='Noisy Data')
    plt.plot(t, clean, 'k--', lw=2, alpha=0.5, label='Ground Truth')
    plt.plot(t, savgol_smoothed, 'r', label='Savitzky-Golay (window=11)')
    plt.plot(t, gaussian_smoothed, 'g', label='Gaussian (σ=2.0)')
    plt.plot(t, ma_smoothed, 'b', label='Moving Average (window=5)')
    
    plt.title("一维步长数据平滑效果对比")
    plt.xlabel("时间步")
    plt.ylabel("步长值")
    plt.legend()
    plt.tight_layout()
    plt.show()

