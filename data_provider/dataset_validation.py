import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path

# 身体连接关系（用于画骨架）
connections = [
    ('kpt1', 'kpt2'),   # Nose -> Left Eye
    ('kpt1', 'kpt3'),   # Nose -> Right Eye
    ('kpt2', 'kpt4'),   # Left Eye -> Left Ear
    ('kpt3', 'kpt5'),   # Right Eye -> Right Ear
    ('kpt6', 'kpt7'),   # Left Shoulder <-> Right Shoulder
    ('kpt6', 'kpt8'),   # Left Shoulder -> Left Elbow
    ('kpt8', 'kpt10'),  # Left Elbow -> Left Wrist
    ('kpt7', 'kpt9'),   # Right Shoulder -> Right Elbow
    ('kpt9', 'kpt11'),  # Right Elbow -> Right Wrist
    ('kpt6', 'kpt12'),  # Left Shoulder -> Left Hip
    ('kpt7', 'kpt13'),  # Right Shoulder -> Right Hip
    ('kpt12', 'kpt13'), # Left Hip <-> Right Hip
    ('kpt12', 'kpt14'), # Left Hip -> Left Knee
    ('kpt14', 'kpt16'), # Left Knee -> Left Ankle
    ('kpt13', 'kpt15'), # Right Hip -> Right Knee
    ('kpt15', 'kpt17')  # Right Knee -> Right Ankle
]

fps = 25
frame_delay = 1000 / fps

video_root = Path("video")
video_root.mkdir(exist_ok=True)

for user_i in range(1, 2):
    # 加载数据文件
    data_path = f"dataset/user_{user_i}.pt"
    try:
        data_tuple = torch.load(data_path)
        keypoints_data, labels = data_tuple[1], data_tuple[2]  # (n,50,35), (n,)
    except FileNotFoundError:
        print(f"File {data_path} not found, skipping user {user_i}")
        continue
    except Exception as e:
        print(f"Error loading {data_path}: {e}")
        continue

    # 为每个任务类型（label 0-9）抽取一个样本
    for task_label in range(10):  # label 对应 task序号-1，所以是0-9
        task_indices = torch.where(labels == task_label)[0]
        
        if len(task_indices) == 0:
            print(f"No samples found for user {user_i}, task {task_label + 1}")
            continue
        
        # 选择第一个样本
        selected_idx = task_indices[0].item()
        selected_keypoints = keypoints_data[selected_idx].numpy()  # (50, 35)
        
        # 将关键点数据转换为DataFrame格式
        # keypoints_ (50, 35) -> 50帧，每帧35个坐标值
        # 假设是17个关键点的x,y坐标（34个值）+ 1个额外值
        num_frames, num_coords = selected_keypoints.shape
        
        # 创建DataFrame，每帧对应17个关键点的x,y坐标
        df_data = []
        num_keypoints = min(17, num_coords // 2)  # 最多17个关键点
        
        for frame in range(num_frames):
            frame_data = {}
            for kpt_idx in range(num_keypoints):
                if kpt_idx * 2 + 1 < num_coords:
                    x_val = selected_keypoints[frame, kpt_idx * 2]
                    y_val = selected_keypoints[frame, kpt_idx * 2 + 1]
                    frame_data[f'kpt{kpt_idx + 1}_x'] = x_val
                    frame_data[f'kpt{kpt_idx + 1}_y'] = y_val
            df_data.append(frame_data)
        
        df = pd.DataFrame(df_data)
        
        # 检查数据是否为空或全部为NaN
        if df.empty or df.isna().all().all():
            print(f"All data is NaN for user {user_i}, task {task_label + 1}, skipping")
            continue
            
        # 提取所有关键点名称（x, y 成对出现）
        kpts = [col.replace('_x', '').replace('_y', '') for col in df.columns if '_x' in col]
        kpts = list(dict.fromkeys(kpts))  # 去重保持顺序

        # 构建索引映射：方便后续查找
        idx_map = {}
        for kpt in kpts:
            idx_map[kpt] = {
                'x': f"{kpt}_x",
                'y': f"{kpt}_y"
            }

        # 计算有效的坐标范围（排除NaN值）
        all_x_vals = []
        all_y_vals = []
        for kpt in kpts:
            x_col = idx_map[kpt]['x']
            y_col = idx_map[kpt]['y']
            x_vals = df[x_col].dropna()
            y_vals = df[y_col].dropna()
            all_x_vals.extend(x_vals.values)
            all_y_vals.extend(y_vals.values)
        
        if len(all_x_vals) == 0 or len(all_y_vals) == 0:
            print(f"No valid coordinates for user {user_i}, task {task_label + 1}, skipping")
            continue
            
        x_min, x_max = min(all_x_vals), max(all_x_vals)
        y_min, y_max = min(all_y_vals), max(all_y_vals)
        
        # 确保范围不为NaN或Inf
        if any(np.isnan([x_min, x_max, y_min, y_max]) | np.isinf([x_min, x_max, y_min, y_max])):
            print(f"Invalid coordinate range for user {user_i}, task {task_label + 1}, skipping")
            continue

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlim(x_min - 0.1, x_max + 0.1)
        ax.set_ylim(y_min - 0.1, y_max + 0.1)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # 图像坐标系Y向下为正，所以翻转Y轴更符合视觉
        ax.set_title(f"Pose Estimation - User {user_i}, Task {task_label + 1}")
        ax.grid(True, linestyle='--', alpha=0.5)

        # 关键点散点图和骨架线对象
        scat = ax.scatter([], [], c='red', s=50, zorder=5)
        lines = {conn: ax.plot([], [], 'b-', linewidth=2)[0] for conn in connections}

        # 动画更新函数
        def update(frame):
            global lines, scat

            # 提取当前帧的所有坐标
            point_coords = {}
            xs = []
            ys = []
            for kpt in kpts:
                x_col = idx_map[kpt]['x']
                y_col = idx_map[kpt]['y']
                x = df.iloc[frame][x_col]
                y = df.iloc[frame][y_col]
                if not (np.isnan(x) or np.isnan(y)):
                    point_coords[kpt] = (x, y)
                    xs.append(x)
                    ys.append(y)
                else:
                    # 如果坐标是NaN，使用前一帧的值或者跳过
                    xs.append(0)  # 或者使用其他默认值
                    ys.append(0)
            
            # 更新关键点位置（只更新非NaN的点）
            if xs and ys:
                scat.set_offsets(np.column_stack((xs, ys)))

            # 更新骨架连线
            for (kpt_a, kpt_b), line_obj in lines.items():
                if kpt_a in point_coords and kpt_b in point_coords:
                    xa, ya = point_coords[kpt_a]
                    xb, yb = point_coords[kpt_b]
                    line_obj.set_data([xa, xb], [ya, yb])
                else:
                    # 如果任一关键点不存在或为NaN，清空连线
                    line_obj.set_data([], [])

            return [scat] + list(lines.values())

        # 创建动画
        ani = FuncAnimation(
            fig,
            update,
            frames=len(df),
            interval=frame_delay,
            blit=True,
            repeat=True
        )

        plt.tight_layout()
        
        # 确保用户目录存在
        user_root = video_root / f'user{user_i}'
        user_root.mkdir(exist_ok=True)
        
        # 保存动画
        output_path = user_root / f'user_{user_i}-task-{task_label + 1}.mp4'
        try:
            ani.save(output_path, writer='ffmpeg', fps=fps)
            print(f"Saved animation: {output_path}")
        except Exception as e:
            print(f"Failed to save {output_path}: {e}")
        
        plt.close(fig)  # 释放内存