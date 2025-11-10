#!/usr/bin/env python3

# 传感器配置
SENSOR_CONFIG = {
    'locations': ['Back'],  # 可选：['Back', 'Neck']
    'modalities': ['ACC_GYRO'],  # 可选：['ACC_GYRO', 'ACC', 'GYRO']
}

# 标签阈值设置
LABEL_THRESHOLD = {
    'annot': 0.6,  # 窗口中有效标注的比例 ≥ 60%
    'main': 0.5    # 窗口中主标签的比例 ≥ 50%
}

# 特征提取配置
FEATURE_CONFIG = {
    'sampling_rate': 100.0,  # Hz
    'window_size': 256,  # samples
}