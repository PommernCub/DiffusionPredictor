# -*- coding: utf-8 -*-
"""
Used for generating training and test sets from simulated data, 
including feature preprocessing

@author: PommernCub
"""
import time
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from utils.preprocess import (
    calc_Euclid_distances, calc_mean_displacement, normalize_data, 
    calc_autocorrelation, power_transform, calc_first_differences,
    calc_gaussian_kde, calc_kde_features, calc_fft_features, 
    plot_process_data                   
    )


# 定义数据目录
train_data_dir = '../simdata0'  # 训练数据目录 (由模拟生成的带标签原始数据)
experiment_dir = '../experiment'  # 实验数据目录
dataset_dir = 'data'  # 生成数据集储存目录
pred_output_dir = 'results'  # 预测结果输出目录
temp_output_dir = 'temp'  # 储存临时预处理文件
if not os.path.exists(dataset_dir):
    os.mkdir(dataset_dir)
if not os.path.exists(pred_output_dir):
    os.mkdir(pred_output_dir)
print(f"生成数据集储存地址：{dataset_dir}")
print(f"预测结果地址：{pred_output_dir}")
# 划分数据集后特征文件名
train_data_outfile = os.path.join(dataset_dir, 
                    "train_data_point_cnn.csv")
test_data_outfile = os.path.join(dataset_dir, 
                    "test_data_point_cnn.csv")
# 读取文件批次大小
BATCH_SIZE = 100
# 异常检测的关键参数
min_region_size = 3  # 连续异常片段的判断最低长度


def stat_global_info(file_list):
    
    displacements = np.array([], dtype=np.float32)
    dist_normal = np.array([], dtype=np.float32)
    dist_abnormal = np.array([], dtype=np.float32)
       
    # 处理每个轨迹文件
    for file_name in tqdm(file_list, desc="读取文件"):
        try:
            data = pd.read_csv(os.path.join(data_dir, file_name))
            displacement, dist_1, dist_0 = calc_Euclid_distances(data)
            del data
            
        except Exception as e:
            print(f"处理异常：{str(e)}")
        
        # 将单个轨迹文件生成的步长拼接
        if len(displacements) == 0:
            displacements = displacement
            dist_normal = dist_1
            dist_abnormal = dist_0
        else:
            displacements = np.concatenate((displacements, displacement))
            dist_normal = np.concatenate((dist_normal, dist_1))
            dist_abnormal = np.concatenate((dist_abnormal, dist_0))

    # 计算数据集整体性质
    # 整体最值, 均值, 标准差; 正常/异常扩散的均值与标准差
    norm_displacements = (displacements - displacements.mean()) / displacements.std()
    stats = {
        'total_mean': displacements.mean(),
        'total_std': displacements.std(),
        'total_max': norm_displacements.max(),
        'total_min': norm_displacements.min(),
        'normal_mean': dist_normal.mean() if len(dist_normal) > 0 else 0,
        'normal_std': dist_normal.std() if len(dist_normal) > 0 else 0,
        'abnormal_mean': dist_abnormal.mean() if len(dist_abnormal) > 0 else 0,
        'abnormal_std': dist_abnormal.std() if len(dist_abnormal) > 0 else 0
    }
    
    # 计算两种标签的高斯核函数
    kde_x0, kde_x1 = calc_gaussian_kde(dist_abnormal, dist_normal)
    
    # 打印总体数据统计
    print(f"共读取文件{len(file_list)}个, 总共轨迹点数目{len(displacements)}个\n")
    ratio0 = len(dist_abnormal)/len(displacements)
    ratio1 = len(dist_normal)/len(displacements)
    print(f"正常扩散占比 {ratio1*100 : .2f}%, 异常扩散占比 {ratio0*100 : .2f}%\n")
    
    return stats, kde_x0, kde_x1


def preprocess_features(stats, file_list, kde_x0, kde_x1):    
# def preprocess_features(stats, file_list):  
    point_features = []
    for file_name in file_list:  # tqdm(file_list, desc="处理数据"):
        try:
            data = pd.read_csv(os.path.join(data_dir, file_name))
            point_id = os.path.splitext(file_name)[0]
            displacement, _, _ = calc_Euclid_distances(data)
            # 归一化步长
            mean, std, xmin, xmax = stats["normal_mean"], stats["normal_std"], stats["total_min"], stats["total_max"]
            norm_dist = normalize_data(displacement, mean, std, xmin, xmax, mode='custom')
            # 获取当前轨迹平均欧式距离 (~平均扩散系数) 与标准差
            mean_distance1, std_distance1 = calc_mean_displacement(norm_dist, 3)
            mean_distance2, std_distance2 = calc_mean_displacement(norm_dist, 5)            
            # 步长窗口自相关系数
            auto_corr_coef = calc_autocorrelation(norm_dist, 5)
            # # 根据整体高斯核函数估算正常/异常概率
            # abnormal_prob, normal_prob = calc_kde_features(kde_x0, kde_x1, norm_dist)
            abnormal_prob, normal_prob = calc_kde_features(kde_x0, kde_x1, mean_distance2)
            # 步长幂变换
            dist_pow = power_transform(mean_distance1)
            # 步长一阶差分
            dist_diff1 = calc_first_differences(norm_dist)
            # 傅里叶fft三阶特征 (特征数量是 window//2+1)
            fft0, fft1, fft2 = calc_fft_features(mean_distance1, 5)

            
            for i in range(len(data)):
                feature = {
                    'point_id': point_id,
                    'frame': i + 1,
                    'displacement': norm_dist[i],  # 2
                    'mean_distance1': mean_distance1[i],  # 3
                    'mean_distance2': mean_distance2[i],  # 4
                    'std_distance1': std_distance1[i],  # 5
                    'std_distance2': std_distance2[i],  # 6
                    'auto_correlation': auto_corr_coef[i],  # 7
                    'normal_prob': normal_prob[i],  # 8
                    'abnormal_prob': abnormal_prob[i],  # 9
                    'distance_pow': dist_pow[i],  # 10
                    'distance_difference': dist_diff1[i],  # 11
                    'fft0': fft0[i],  # 12
                    'fft1': fft1[i],  # 13
                    'fft2': fft2[i],  # 14
                    'is_normal': abs(data['Diffusion Coefficient'].iloc[i] - 1.0) < 0.01,
                    'X': data['X'].iloc[i],
                    'Y': data['Y'].iloc[i],
                    'Diffusion Coefficient': data['Diffusion Coefficient'].iloc[i]
                }
                point_features.append(feature)
            
            del data
            
        except Exception as e:
            print(f"处理异常：{str(e)}")
    
    return point_features


def merge_batch_results(tmp_dir):
    """合并临时批次文件"""
    batch_files = [f for f in os.listdir(tmp_dir) if f.startswith('batch_')]
    dfs = []
    
    for f in tqdm(batch_files, desc="合并批次"):
        df = pd.read_csv(os.path.join(tmp_dir, f))
        dfs.append(df)
        os.remove(os.path.join(tmp_dir, f))  # 清理临时文件
        
    return pd.concat(dfs, ignore_index=True)
    
    
def split_dataset(processed_df):
    print("划分训练/测试集……")
    # 划分训练测试集
    unique_trajectories = processed_df['point_id'].unique()
    train_trajectories, test_trajectories = train_test_split(
        unique_trajectories,
        test_size=0.2,
        random_state=42
    )
    
    train_data = processed_df[processed_df['point_id'].isin(train_trajectories)]
    test_data = processed_df[processed_df['point_id'].isin(test_trajectories)]
    
    train_data.to_csv(train_data_outfile, index=False)
    test_data.to_csv(test_data_outfile, index=False)    
    print(f"生成训练集{train_data_outfile}\n生成验证集{test_data_outfile}")
    
    return train_data, test_data



if __name__ == "__main__":

    # 加载原始数据文件
    data_dir = train_data_dir
    # 给文件名排序，读取更快
    file_list = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')], \
                       key=lambda x: int(x.split('_')[-1].split('.')[0]))
    file_list = file_list[:10000]  # 取前10000个数据作为样本
    print("开始加载数据...")
    
    start_time = time.time()
    # 全局性质统计
    stats, kde_x0, kde_x1 = stat_global_info(file_list)
    # 数据分批量预处理
    os.makedirs(temp_output_dir, exist_ok=True)  # 存储临时预处理文件
    for batch_idx in tqdm(range(0, len(file_list), BATCH_SIZE), desc="处理批次"):
        batch_files = file_list[batch_idx:batch_idx+BATCH_SIZE]
        # 特征计算
        batch_features = preprocess_features(stats, batch_files, kde_x0, kde_x1)
        # batch_features = preprocess_features(stats, batch_files)
        # 保存本批次结果
        pd.DataFrame(batch_features).to_csv(
            os.path.join(temp_output_dir, f"batch_{batch_idx//BATCH_SIZE}.csv"),
            index=False
        )
    
    # 合并所有批次
    processed_df = merge_batch_results(temp_output_dir)
    os.rmdir(temp_output_dir)  # 删除空的临时文件夹
    print(f"\n数据预处理耗时: {time.time() - start_time:.2f} 秒")
    # 划分数据集
    split_dataset(processed_df)