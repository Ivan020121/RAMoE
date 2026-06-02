import random
import pickle
import numpy as np
from collections import defaultdict

import torch


def fix_random_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_and_split_data(data_path, balance=True):
    """
    Args:
        data_path: pkl 文件路径
        balance: 是否执行类别平衡 (默认 True)
    """
    print(f"🚀 正在加载数据集: {data_path}")
    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)

    # --- 1. 标签映射 (Sit/Stand -> Idle) ---
    raw_activities = sorted(list(dataset.keys()))
    merge_map = {}
    for act in raw_activities:
        if act in ['Sit', 'Stand']:
            merge_map[act] = 'Idle'
        else:
            merge_map[act] = act

    unique_new_activities = sorted(list(set(merge_map.values())))
    print(unique_new_activities)
    act_to_id = {name: i for i, name in enumerate(unique_new_activities)}
    id_to_act = {i: name for name, i in act_to_id.items()}

    print(f"📋 标签映射: {act_to_id}")

    # --- 2. 按类别分组 (Grouping) ---
    grouped_data = defaultdict(lambda: {'v': [], 'i': []})

    for act, data in dataset.items():
        videos = data['video']
        imus = data['imu']
        if len(videos) == 0: continue

        new_act_name = merge_map[act]
        label_id = act_to_id[new_act_name]

        grouped_data[label_id]['v'].append(videos)
        grouped_data[label_id]['i'].append(imus)

    # --- 3. 🧹 NaN 清洗 (Cleaning) ---
    print("\n🧹 开始检查并移除 NaN 样本...")
    total_nan_removed = 0

    # 遍历每个类别进行合并和清洗
    for lid in list(grouped_data.keys()):
        # 先合并该类别的所有数据
        v_data = np.concatenate(grouped_data[lid]['v'], axis=0)  # (N, 3, 30, 17, 2)
        i_data = np.concatenate(grouped_data[lid]['i'], axis=0)  # (N, 100, 3)

        N = v_data.shape[0]

        # 检测 Video 是否含 NaN (将除N以外的维度展平检测)
        # any(axis=1) 表示只要该样本里有一个数值是NaN，该样本就标记为True
        nan_mask_v = np.isnan(v_data.reshape(N, -1)).any(axis=1)

        # 检测 IMU 是否含 NaN
        nan_mask_i = np.isnan(i_data.reshape(N, -1)).any(axis=1)

        # 取并集：任意模态坏了，整个样本就坏了
        bad_sample_mask = nan_mask_v | nan_mask_i
        num_nans = np.sum(bad_sample_mask)

        if num_nans > 0:
            total_nan_removed += num_nans
            # 只保留 False (即没有NaN) 的样本
            clean_mask = ~bad_sample_mask

            grouped_data[lid]['v'] = v_data[clean_mask]
            grouped_data[lid]['i'] = i_data[clean_mask]

            print(f"   ⚠️ 类别 '{id_to_act[lid]}': 移除了 {num_nans} 个包含 NaN 的样本 (剩余 {np.sum(clean_mask)})")
        else:
            # 如果没有 NaN，直接保存 numpy 数组
            grouped_data[lid]['v'] = v_data
            grouped_data[lid]['i'] = i_data

    if total_nan_removed == 0:
        print("   ✅ 完美！数据集中没有发现 NaN。")
    else:
        print(f"   🗑️ 总计移除了 {total_nan_removed} 个坏样本。")

    # --- 4. 类别平衡 (Balancing) ---
    if balance:
        # 基于清洗后的数据计算最小数量
        counts = [len(grouped_data[lid]['v']) for lid in grouped_data]
        if len(counts) == 0 or min(counts) == 0:
            raise ValueError("❌ 某个类别在清洗后变为空，无法进行平衡，请检查数据质量！")

        min_count = min(counts)
        print(f"\n⚖️ 执行类别平衡 (对齐至 {min_count} 样本)...")

        final_v, final_i, final_y = [], [], []

        for lid in sorted(grouped_data.keys()):
            v_data = grouped_data[lid]['v']
            i_data = grouped_data[lid]['i']

            # 随机抽样
            indices = np.random.choice(len(v_data), min_count, replace=False)

            final_v.append(v_data[indices])
            final_i.append(i_data[indices])
            final_y.append(np.full(min_count, lid))

        X_video = np.concatenate(final_v, axis=0)
        X_imu = np.concatenate(final_i, axis=0)
        y = np.concatenate(final_y, axis=0)
    else:
        # 不执行全局平衡，先把各类别数据拼接起来供训练使用
        print("\n⚠️ 跳过类别平衡...")
        X_video = np.concatenate([grouped_data[lid]['v'] for lid in grouped_data], axis=0)
        X_imu = np.concatenate([grouped_data[lid]['i'] for lid in grouped_data], axis=0)
        y = np.concatenate([np.full(len(grouped_data[lid]['v']), lid) for lid in grouped_data], axis=0)

    # --- 5. 维度变换 (Transpose) ---

    # --- 5. 维度变换 (Transpose) ---
    # 原始: (N, 3, 30, 17, 2) -> (N, View, Time, Joint, Coord)
    # 目标: (N, 2, 30, 17, 3) -> (N, Coord, Time, Joint, View)
    X_video = np.transpose(X_video, (0, 4, 2, 3, 1))

    print(f"\n📦 最终 Tensor 形状:")
    print(f"   Video: {X_video.shape}")
    print(f"   IMU:   {X_imu.shape}")
    print(f"   Label: {y.shape}")

    # --- 6. 转 Tensor & Split ---
    X_video = torch.tensor(X_video, dtype=torch.float32)
    X_imu = torch.tensor(X_imu, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    # 全局打乱
    indices = np.arange(len(y))
    np.random.shuffle(indices)

    # 计算切分点
    n_total = len(y)
    n_train = int(n_total * 0.6)
    n_val = int(n_total * 0.8)

    # 提取
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_val]
    test_idx = indices[n_val:]

    train_set = (X_imu[train_idx], X_video[train_idx], y[train_idx])
    val_set = (X_imu[val_idx], X_video[val_idx], y[val_idx])
    test_set = (X_imu[test_idx], X_video[test_idx], y[test_idx])

    # 如果不做全局平衡, 仍需确保验证集/测试集内部各类别样本数量一致
    if not balance:
        def balance_subset(subset):
            x_i, x_v, label = subset
            # label may be tensor
            labels_np = label.numpy()
            unique, counts = np.unique(labels_np, return_counts=True)
            if len(unique) <= 1:
                return subset
            min_cnt = counts.min()
            selected_indices = []
            for lid in unique:
                lid_idx = np.where(labels_np == lid)[0]
                if len(lid_idx) > min_cnt:
                    lid_idx = np.random.choice(lid_idx, min_cnt, replace=False)
                selected_indices.append(lid_idx)
            selected_indices = np.concatenate(selected_indices)
            np.random.shuffle(selected_indices)
            return (x_i[selected_indices], x_v[selected_indices], label[selected_indices])

        val_set = balance_subset(val_set)
        test_set = balance_subset(test_set)

    print(f"\n✂️ 数据划分 (Train/Val/Test): {len(train_set[2])} / {len(val_set[2])} / {len(test_set[2])}")

    return train_set, val_set, test_set


if __name__ == '__main__':
    fix_random_seed(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    batch_size = 128
    seq_len = 100
    n_fft = 64
    hop_length = 8
    in_channels = 3
    patch_size = 16
    stride = 16
    depth = 12
    num_classes = 8
    out_channels = 32
    graph_args = {"layout": "openpose", "strategy": "spatial"}
    edge_importance_weighting = True
    lr = 0.001
    epoch = 300
    save_path = './'
    data_path = "multimodal_dataset.pkl"

    train_set, val_set, test_set = load_and_split_data(data_path, balance=False)
    torch.save((train_set, val_set, test_set), "newdataset_unbalance.pt")