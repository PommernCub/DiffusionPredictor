import os
import pandas as pd
import numpy as np
import csv
import bisect
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from collections import defaultdict, Counter


def find_continuous_segments(column):
    """识别连续值段"""
    segments = []
    if len(column) == 0:
        return segments
    
    current_val = column[0]
    start_idx = 0
    
    for i in range(1, len(column)):
        if column[i] != current_val:
            segments.append((start_idx, i-1, current_val))
            current_val = column[i]
            start_idx = i
    segments.append((start_idx, len(column)-1, current_val))
    return segments


def process_column(column, process=True):
    """处理单个列的不连续段"""
    # 首先将非1的值转换为0 (column == 1? Y->1, N->0)
    column = np.where(column == 1, 1, 0)
    new_col = column.copy()
    
    if process:
        segments = find_continuous_segments(column)

        # 预计算所有需要替换的区间
        replacements = []
        for seg in segments:
            start, end, val = seg
            seg_length = end - start + 1
            
            if seg_length >= 3:
                continue  # 跳过有效段
            
            # 计算上下文窗口范围
            ctx_start = max(0, start - 3)
            ctx_end = min(len(column)-1, end + 3)
            context = column[ctx_start:ctx_end+1]
            
            # 统计上下文中的主要值
            count0 = np.sum(context == 0)
            count1 = np.sum(context == 1)
            new_val = 0 if count0 > count1 else 1 if count1 > count0 else val
            
            replacements.append((start, end, new_val))
        
        # 应用替换（避免修改影响后续判断）
        for start, end, new_val in replacements:
            new_col[start:end+1] = new_val
    
    return 1 - new_col  # 倒置0和1（0定义为自由扩散，1代表吸附发生）


def extract_segments(labels):
    """提取连续1的段落及其起止索引"""
    segments = []
    start = None
    for idx, val in enumerate(labels):
        if val == 1:
            if start is None:
                start = idx
        else:
            if start is not None:
                segments.append((start, idx-1))
                start = None
    if start is not None:  # 处理末尾连续的1
        segments.append((start, len(labels)-1))
    return segments


def Jaccard_index(true_segments, predicted_segments):
    """计算每个真实片段与预测片段的最佳Jaccard指数，并返回匹配的预测片段长度"""
    # 预处理步骤：按起始点排序预测区间
    predicted_sorted = sorted(predicted_segments, key=lambda x: x[0])
    p_starts = [seg[0] for seg in predicted_sorted]
    p_ends = [seg[1] for seg in predicted_sorted]
    
    # 转换为 numpy 数组以加速计算
    p_starts_np = np.array(p_starts, dtype=np.int64)
    p_ends_np = np.array(p_ends, dtype=np.int64)
    
    jaccard_indices = []
    matched_pred_lengths = []  # 存储匹配的预测片段长度
    
    for t_start, t_end in true_segments:
        # 步骤1：快速定位可能重叠的预测区间
        high = bisect.bisect_right(p_starts, t_end)
        
        # 步骤2：向量化筛选有效区间
        if high == 0:
            jaccard_indices.append(0.0)
            matched_pred_lengths.append(0)  # 无匹配，长度为0
            continue        
        # 提取候选区间
        candidate_ends = p_ends_np[:high]
        mask = candidate_ends >= t_start        
        # 无有效候选则跳过
        if not mask.any():
            jaccard_indices.append(0.0)
            matched_pred_lengths.append(0)  # 无匹配，长度为0
            continue
        
        # 步骤3：向量化计算交集和并集
        valid_starts = p_starts_np[:high][mask]
        valid_ends = candidate_ends[mask]
        
        inter_start = np.maximum(t_start, valid_starts)
        inter_end = np.minimum(t_end, valid_ends)
        inter_len = inter_end - inter_start + 1
        
        union_start = np.minimum(t_start, valid_starts)
        union_end = np.maximum(t_end, valid_ends)
        union_len = union_end - union_start
        
        # 步骤4：批量计算 Jaccard 指数
        ji_values = inter_len / (union_len + 1e-9)  # 防止除零
        max_ji_index = np.argmax(ji_values)
        max_ji = np.max(ji_values)
        # 获取匹配的预测片段长度
        matched_start = valid_starts[max_ji_index]
        matched_end = valid_ends[max_ji_index]
        matched_length = matched_end - matched_start + 1
        matched_pred_lengths.append(matched_length)
        
        # 提前终止条件：找到完美匹配
        if max_ji >= 1.0 - 1e-6:
            jaccard_indices.append(1.0)
        else:
            jaccard_indices.append(max_ji)    
    return jaccard_indices, matched_pred_lengths


def save_segment_lengths(dir_path, true_dist, predicted_dist, grouped_stats):
    """保存段落长度分布到文件"""
    output_path = os.path.join(dir_path, "segments_dist.csv")
    # 合并所有可能的长度值
    all_lengths = set(true_dist.keys()).union(predicted_dist.keys())
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Segment_Length", "True_Count", "Predicted_Count",
                         "Pred_Mean", "Pred_Std", "Jaccard_Index"])
        
        for length in sorted(all_lengths):
            # 获取该长度分组的统计信息
            stats = grouped_stats.get(length, {"pred_mean": 0, "pred_std": 0, "ji_mean": 0})
            writer.writerow([
                length,
                true_dist.get(length, 0),
                predicted_dist.get(length, 0),
                stats["pred_mean"],
                stats["pred_std"],
                stats["ji_mean"]
            ])
    print(f"段落长度分布已输出到 {output_path}")


def save_ji_distribution(dir_path, ji_dist, total_n):
    """保存JI指数分布到文件（带频率）"""
    output_path = os.path.join(dir_path, "JI_dist.csv")
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["JI_Value", "Frequency"])
        # 按JI值排序并计算频率
        for ji_val in sorted(ji_dist.keys()):
            count = ji_dist[ji_val]
            frequency = count / total_n if total_n > 0 else 0.0
            writer.writerow([
                ji_val,
                round(frequency, 4)  # 频率保留4位小数
            ])
    print(f"Jaccard Index 分布已输出到 {output_path}")


def iqr_filter(data, q1=25, q3=75, k=1.5):
    """
    IQR异常值过滤函数
    参数：
        data：输入数组
        q1：下四分位数百分比（默认25%）
        q3：上四分位数百分比（默认75%）
        k：IQR倍数（默认1.5）
    返回：
        过滤后的数据数组
    """
    q1_val = np.percentile(data, q1)
    q3_val = np.percentile(data, q3)
    iqr = q3_val - q1_val
    lower_bound = q1_val - k * iqr
    upper_bound = q3_val + k * iqr
    
    return data[(data >= lower_bound) & (data <= upper_bound)]


def dataprocess(dir_path, input_file):
    # 读取数据
    df = pd.read_csv(os.path.join(dir_path, input_file))
    # 处理两列数据
    df['true_label_processed'] = process_column(df.iloc[:, 3].values, True)
    df['predicted_label_processed'] = process_column(df.iloc[:, 2].values, True)
   
    # 保存结果
    output_path = os.path.join(dir_path, 'processed-results.csv')
    df.to_csv(output_path, index=False)    
    print(f"处理后数据结果保存在 {output_path}")
    # 计算评估指标
    y_true = df['true_label_processed']
    y_pred = df['predicted_label_processed']    
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred)
    }
    # for k, v in metrics.items():
    #     print(f"{k}: {v:.4f}")
    print(f"""\n经过后处理, 模型性能评估：
                    Accuracy:  {metrics['Accuracy']:.4f}
                    Precision: {metrics['Precision']:.4f}
                    Recall:    {metrics['Recall']:.4f}
                    F1 Score:  {metrics['F1']:.4f}"""
                    )
    
    # 统计连续段落数
    true_segments = extract_segments(y_true)
    predicted_segments = extract_segments(y_pred)
    N_actual = len(true_segments)
    N_pred = len(predicted_segments)
    print(f"真实标签连续段落数 N_actual={N_actual}")
    print(f"预测标签连续段落数 N_pred={N_pred}\n")
    
    # 统计段落长度分布
    true_lengths = [end-start+1 for (start, end) in true_segments]
    predicted_lengths = [end-start+1 for (start, end) in predicted_segments]
    true_dist = dict(Counter(true_lengths))
    predicted_dist = dict(Counter(predicted_lengths))
    # save_segment_lengths(dir_path, true_dist, predicted_dist)
    
    # 计算Jaccard指数
    jaccard_indices, matched_pred_lengths = Jaccard_index(true_segments, predicted_segments)
    average_ji = sum(jaccard_indices)/len(jaccard_indices) if jaccard_indices else 0
    ji_dist = dict(Counter(round(ji, 2) for ji in jaccard_indices))  # 保留两位小数
    print(f"平均Jaccard指数: {average_ji:.4f}")
    save_ji_distribution(dir_path, ji_dist, len(jaccard_indices))

    # 按真实片段长度分组，收集匹配的预测长度和JI值
    length_group = defaultdict(lambda: {"pred_lengths": [], "ji_values": []})    
    for i, length in enumerate(true_lengths):
        length_group[length]["pred_lengths"].append(matched_pred_lengths[i])
        length_group[length]["ji_values"].append(jaccard_indices[i])    
    # 计算每个长度分组的统计量
    grouped_stats = {}
    for length, data in length_group.items():
        pred_lengths = data["pred_lengths"]
        ji_values = data["ji_values"]        
        # 计算预测长度的平均值和标准差
        if pred_lengths:
            # 采用四分位数方法去掉异常值
            pred_lengths = iqr_filter(np.array(pred_lengths), k=0.5)
            pred_mean = np.mean(pred_lengths)
            pred_std = np.std(pred_lengths)
        else:
            pred_mean = pred_std = 0
        # 计算JI平均值
        ji_mean = np.mean(ji_values) if ji_values else 0        
        grouped_stats[length] = {
            "pred_mean": round(pred_mean, 4),
            "pred_std": round(pred_std, 4),
            "ji_mean": round(ji_mean, 4)
        }
    
    # 保存包含新增统计量的长度分布
    save_segment_lengths(dir_path, true_dist, predicted_dist, grouped_stats)
    


if __name__ == "__main__":
    SIZE = 30  # 输入文件名中粒子尺寸
    MODEL = 'cnn'
    dir_path = "results\\exp1"
    input_file = "test-results.csv"
    dataprocess(dir_path, input_file)
