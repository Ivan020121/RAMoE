import math
import logging
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm


DATA_ROOT = Path("/rd/zljx/dataset/SWIT/SWIT_Dataset/")
IMU_PATH = DATA_ROOT / "IMU_Sensor_Raw"
KEYPOINTS_PATH = DATA_ROOT / "Keypoints_Raw"
SEGMENT_PATH = DATA_ROOT / "Segmentation_Status"
USER_NUM = 27
TASK_NUM = 10

class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(TqdmLoggingHandler())

def divide(raw_df, start_time, end_time, data_type, step:float=10, segment_length=200) -> np.ndarray:
    """从[start_time, end_time)中进行采样长度为segment_length，步长为step的采样"""
    segment_df = raw_df[(raw_df['Time'] >= start_time) & (raw_df['Time'] <= end_time)]
    window_data = []
    for start_idx in [math.floor(i) for i in np.arange(0.0, len(segment_df) - segment_length + 1, step)]:
        segment = segment_df.iloc[start_idx : start_idx + segment_length]
        if data_type == 'imu':
            data = segment[['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'Mx', 'My', 'Mz']].values
        elif data_type == 'kp':
            segment = segment.drop(columns=['Time'], errors='ignore')
            data = segment.values  
        window_data.append(data)
    # 标注的start time、end time比总时长还大
    if not window_data:
        logger.warning(f"Invalid segmentation，time too big：start_time: {start_time}, end_time: {end_time}")
        return None
    return np.stack(window_data, axis=0)

def process_single_task_for_user(task_i, user_i, size=None, step=10, padding_time=1.0, segment_length=200, nan_tolerance=False):
    raw_imu = pd.read_csv(IMU_PATH / f'Task_{task_i}_User_{user_i}.csv')
    raw_kp = pd.read_csv(KEYPOINTS_PATH / f'Task_{task_i}_User_{user_i}_kypt.csv')
    raw_kp['Time'] = raw_kp.index * 0.04
    # 对任务1-9和任务10分别处理，只有任务10才提供size

    if not size:
        seg = pd.read_csv(SEGMENT_PATH / f'ST_Task_{task_i}.csv')
        user_starts_ends = seg[f'User_{user_i}'].dropna().values
        adjusted_times = [max(0, (time * 0.01) - padding_time) if idx % 2 == 0 else max(2, (time * 0.01) + padding_time) for idx, time in enumerate(user_starts_ends)]
        imu_list = []
        kp_list = []
        for i in range(0, len(adjusted_times), 2):
            imu_seg = divide(raw_imu, adjusted_times[i], adjusted_times[i + 1], data_type='imu', step=step, segment_length=segment_length)
            kp_seg = divide(raw_kp, adjusted_times[i], adjusted_times[i + 1], data_type='kp', step=step/4, segment_length=segment_length//4)
            
            if imu_seg is None or kp_seg is None:
                continue
            if not(imu_seg.size and kp_seg.size):
                continue


            align_size = min(imu_seg.shape[0], kp_seg.shape[0])
            imu_seg = imu_seg[:align_size]
            kp_seg = kp_seg[:align_size]

            if not nan_tolerance and np.isnan(kp_seg).any():
                mask = ~np.isnan(kp_seg).any(axis=(1, 2))
                imu_seg = imu_seg[mask]
                kp_seg = kp_seg[mask]

            imu_list.append(imu_seg)
            kp_list.append(kp_seg)
    else:
        duration = len(raw_imu)
        # duration=step*(size-1)+size，整除是为了便于对齐，不完全按公式是因为会因为精度问题造成数目减少，需要留余量
        imu_list: list[np.ndarray] = [divide(raw_imu, padding_time, duration-padding_time, data_type='imu',step=(duration)//(size+10), segment_length=segment_length)]
        kp_list: list[np.ndarray] = [divide(raw_kp, padding_time, duration-padding_time, data_type='kp', step=(duration)//(size+10)/4, segment_length=segment_length//4)]
           
    
    imu_X = np.concatenate(imu_list, axis=0)
    kp_X = np.concatenate(kp_list, axis=0)
    y = np.full(imu_X.shape[0], task_i-1)
    return imu_X, kp_X, y

def balance_classes(user_imu_X_list, user_kp_X_list, y_list, task_i):
    combined_imu = np.concatenate(user_imu_X_list, axis=0)
    combined_kp = np.concatenate(user_kp_X_list, axis=0)
    combined_y = np.concatenate(y_list, axis=0)

    counter = Counter(combined_y)
    min_count = min(counter.values())

    balanced_imu, balanced_kp, balanced_y = [], [], []

    for label in counter.keys():
        indices = np.where(combined_y == label)[0]
        selected_indices = np.random.choice(indices, min_count, replace=False)

        balanced_imu.append(combined_imu[selected_indices])
        balanced_kp.append(combined_kp[selected_indices])
        balanced_y.append(combined_y[selected_indices])
    

    balanced_imu = np.concatenate(balanced_imu, axis=0)
    balanced_kp = np.concatenate(balanced_kp, axis=0)
    balanced_y = np.concatenate(balanced_y, axis=0)

    

    return balanced_imu, balanced_kp, balanced_y

with tqdm(total=TASK_NUM*(USER_NUM-1)) as pbar:
    for user_i in [i for i in range(1, USER_NUM+1) if i!=16]:
        user_imu_X_list, user_kp_X_list, user_y_list = [], [], []

        for task_i in range(1, TASK_NUM+1):
            pbar.set_description(f"User{user_i}-Task{task_i}: ")
            min_size = None
            if task_i==10:
                min_size = min([i.shape[0] for i in user_imu_X_list])
            imu_X, kp_X, y = process_single_task_for_user(task_i, user_i, min_size, step=10, padding_time=1, segment_length=200, nan_tolerance=False)
            user_imu_X_list.extend([imu_X])
            user_kp_X_list.extend([kp_X])
            user_y_list.extend([y])
            pbar.update(1)
        
        
        if user_imu_X_list and user_y_list:
            balanced_user_imu_X, balanced_user_kp_X, balanced_user_y = balance_classes(user_imu_X_list, user_kp_X_list, user_y_list, task_i)
            IMU_X_array = np.concatenate([balanced_user_imu_X], axis=0)
            KP_X_array = np.concatenate([balanced_user_kp_X], axis=0)
            y_array = np.concatenate([balanced_user_y], axis=0)
            #print(f"All class label count - user_{user_i}: {np.bincount(y_array)}")
            torch.save((torch.tensor(IMU_X_array, dtype=torch.float32), 
                        torch.tensor(KP_X_array, dtype=torch.float32), 
                        torch.tensor(y_array, dtype=torch.long)), 
                        f'dataset/{f'user_{user_i}'}.pt'
            )





