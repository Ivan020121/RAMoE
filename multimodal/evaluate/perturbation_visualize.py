from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import torch

from model.multimodal.moe import PRISM
from multimodal.train.base import get_unimodal_models
from util.torch_utils import (
    augment_saved_dataset,
    fix_random_seed,
    load_and_split_data,
)
from config import config

def imu_vis(data, channel_names, title_suffix):
    # 创建9通道垂直排列的子图
    row = data.shape[1]
    fig, axes = plt.subplots(row, 1, figsize=(8, 6), sharex=True)
    
    for ch in range(row):
        axes[ch].plot(data[:, ch], linewidth=1.2, color=f'C{ch}')
        axes[ch].set_ylabel(channel_names[ch], fontsize=9 , rotation=0, labelpad=20,va='center')
        # axes[ch].grid(True, linestyle='--', alpha=0.7)
        axes[ch].tick_params(axis='both', labelsize=8)
        axes[ch].set_yticks([])
    
    # 公共设置
    axes[0].set_title(f'IMU Signal ({title_suffix})', fontsize=14, pad=15)
    axes[-1].set_xlabel('Time Step', fontsize=10)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0)
    # plt.tight_layout(pad=2.0, h_pad=0.5)
    
    # 保存图像
    save_path = f"tmp/imu_{title_suffix}.svg"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight',format="svg")
    plt.close(fig)
    print(f"Saved visualization: {save_path}")

def normalize_keypoints(kps, img_width, img_height):
    kps = np.array(kps, dtype=np.float32)
    kps_norm = np.zeros_like(kps)
    kps_norm[:, 0] = (kps[:, 0] / img_width) * 2 - 1  # x: [0,w] -> [-1,1]
    kps_norm[:, 1] = (kps[:, 1] / img_height) * 2 - 1 # y: [0,h] -> [-1,1]
    return kps_norm

def kp_vis(kps, title_suffix):
    connections_idx = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 绘制骨骼连接（蓝色）
    for i, j in connections_idx:
        if 0 <= i < len(kps) and 0 <= j < len(kps):
            x = [kps[i, 0], kps[j, 0]]
            y = [kps[i, 1], kps[j, 1]]
            ax.plot(x, y, 'b-', linewidth=2.5, alpha=0.8, zorder=2)
    
    # 绘制关键点（红色）
    ax.scatter(kps[:, 0], kps[:, 1], 
        c='red', s=120, edgecolors='white', 
        linewidth=1.5, zorder=3, label='Keypoints')
    
    # 设置坐标轴
    ax.set_xlim(-0.5, 2)
    ax.set_ylim(-0.5, 2)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # 匹配图像坐标系（y向下为正）
    ax.grid(True, linestyle='--', alpha=0.6, zorder=0)
    ax.set_xlabel('X (Normalized)', fontsize=16)
    ax.set_ylabel('Y (Normalized)', fontsize=16)
    ax.set_title('Human Pose Keypoints', fontsize=18)
    
    # 添加图例
    ax.legend(loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    save_path = f"tmp/kp_{title_suffix}.svg"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight',format="svg")
    plt.close(fig)
    print(f"Saved visualization: {save_path}")

if __name__ == "__main__":
    logger = config.logger
    seed = 3407
    imu_channels = 9
    train_epoch = 40
    repeat_train = 1
    checkpoint_path = Path("checkpoint/use")
    device = "cuda:1"
    batch_size = 256
    lr = 6e-5
    model_name = "PRISM"

    logger.info("seed = %s", seed)
    fix_random_seed(seed, True)
    train_set, val_set, test_set = load_and_split_data(
        Path("dataset"),
        imu_channel=imu_channels,
        load_dataset_path=Path("dataset/dataset_ag.pt"),
    )

    modal = "multimodal"
    # imu 96, kp 120
    sample_idx = 120

    imu_encoder, kp_encoder = get_unimodal_models(
        train_set,
        val_set,
        test_set,
        device=device,
        no_train=True,
        with_head=True,
        return_both=True,
    )

    model = PRISM(imu_encoder, kp_encoder, 512, dropout=0.3, device=device)
    model.load_state_dict(
        torch.load(checkpoint_path / "PRISM 2026-02-01 22:21:27-latest.pt")
    )

    sample_set = (test_set[0][sample_idx,:].unsqueeze(0), test_set[1][sample_idx,:].unsqueeze(0), test_set[2][sample_idx].unsqueeze(0))
    imu_names = [
        'Acc-X', 'Acc-Y', 'Acc-Z',
        'Gyr-X', 'Gyr-Y', 'Gyr-Z',
        'Mag-X', 'Mag-Y', 'Mag-Z'
    ]
    # imu_vis(sample_set[0][0,:], imu_names, "Origin")
    # kp = sample_set[1][0,:].permute(1,2,0)[0, :]
    # kp_vis(kp, "Origin")
    
    augment_imu_sample = []
    channel_names=[]
    item = "Translation"
    for i in range(15, 16):
        channel_names.append(f"{0.1 * i:.1f}")
        augment_test_set = augment_saved_dataset(
            sample_set,
            {
                "augment_ratio": 1,
                "only_augment": True,
                "imu": {
                    # "rotation": {"deg": i * 10},
                    # "scale": {"min": 1 - 0.1 * i, "max": 1 + 0.1 * i},
                    # "jitter": {"sigma": 0.1 * i},
                    # "augment_ratio": 1,
                },
                "kp": {
                    # "scale": {"min": 1 + 0.1 * i, "max": 1 + 0.1 * i},
                    "translation": {"max": 0.1 * i},
                    "augment_ratio": 1,
                },
            },
        )
        imu, kp, label = augment_test_set[0][0,:], augment_test_set[1][0,:], augment_test_set[2][0]
        
        # imu_vis(imu, imu_names, f"{item}-{i}")
        kp = kp.permute(1,2,0)[0, :]
        kp_vis(kp, f"{item}-{i}")
        # augment_imu_sample.append(imu[:, 0])

    # imu_vis(torch.stack(augment_imu_sample, dim=1), channel_names, item)


