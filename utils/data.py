"""Original augmentation, person splitting and tensor loaders."""
from collections import defaultdict
import logging
import math
from pathlib import Path
import random
from typing import List, Literal, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import config


def apply_rotation_to_sample(x: torch.Tensor, imu_cfg: dict):
    deg = float(imu_cfg.get("deg", 0.0))
    max_rad = float(deg) * math.pi / 180.0
    num_triplets = x.shape[1] // 3
    angle = float(torch.empty(1, device=x.device).uniform_(-max_rad, max_rad).item())
    axis = torch.randn(3, device=x.device, dtype=x.dtype)
    axis = axis / (axis.norm() + 1e-8)
    ux, uy, uz = axis.tolist()
    c = math.cos(angle)
    s = math.sin(angle)
    R = torch.tensor(
        [
            [
                c + ux * ux * (1 - c),
                ux * uy * (1 - c) - uz * s,
                ux * uz * (1 - c) + uy * s,
            ],
            [
                uy * ux * (1 - c) + uz * s,
                c + uy * uy * (1 - c),
                uy * uz * (1 - c) - ux * s,
            ],
            [
                uz * ux * (1 - c) - uy * s,
                uz * uy * (1 - c) + ux * s,
                c + uz * uz * (1 - c),
            ],
        ],
        device=x.device,
        dtype=x.dtype,
    )
    for t_idx in range(num_triplets):
        i0 = t_idx * 3
        v = x[:, i0 : i0 + 3]  # (T,3)
        x[:, i0 : i0 + 3] = torch.matmul(v, R.transpose(0, 1))
    return x


def merge_data(
    data_list: List[Tuple],
    apply_augmentation: bool = False,
    augment: dict = {},
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    imus, kps, labels = zip(*data_list)
    merged_imus, merged_kps_raw, merged_labels = (
        torch.cat(imus),
        torch.cat(kps),
        torch.cat(labels),
    )

    if apply_augmentation and augment:
        total_aug_ratio = float(augment.get("augment_ratio", 0.0))
        augment_only = augment.get("only_augment", False)
        if total_aug_ratio > 0:
            N = merged_imus.shape[0]
            n_total_aug = int(round(total_aug_ratio * N))
            n_total_aug = max(0, min(n_total_aug, N))
            if n_total_aug > 0:
                # 随机选择要增强的样本
                perm = torch.randperm(N, device=merged_imus.device)[:n_total_aug]
                # 复制样本作为增强基础
                if augment_only:
                    if n_total_aug == N:
                        aug_imus = merged_imus
                        aug_kps_raw = merged_kps_raw
                        aug_labels = merged_labels

                    aug_imus = merged_imus[perm]
                    aug_kps_raw = merged_kps_raw[perm]
                    aug_labels = merged_labels[perm]
                else:
                    aug_imus = merged_imus[perm].clone()
                    aug_kps_raw = merged_kps_raw[perm].clone()
                    aug_labels = merged_labels[perm].clone()

                # 2. 获取模态特定配置
                imu_cfg = augment.get("imu", {}) or {}
                kp_cfg = augment.get("kp", {}) or {}

                # 3. 为增强样本生成独立增强掩码
                do_imu_aug = torch.rand(n_total_aug, device=merged_imus.device) < float(
                    imu_cfg.get("augment_ratio", 0.0)
                )
                do_kp_aug = torch.rand(n_total_aug, device=merged_imus.device) < float(
                    kp_cfg.get("augment_ratio", 0.0)
                )

            if imu_cfg and do_imu_aug.any():
                # 提取需要IMU增强的样本索引
                imu_indices = torch.where(do_imu_aug)[0]
                for idx in imu_indices:
                    x = aug_imus[idx]  # (T, C_imu)
                    T, _ = x.shape
                    jcfg = imu_cfg.get("jitter", {})
                    if jcfg:
                        sigma = float(jcfg.get("sigma", 0.01))
                        signal_std = x.std(dim=0, keepdim=True)
                        x = x + torch.randn_like(x) * (sigma * signal_std)

                    # scale: multiply by a scalar (constant or random range)
                    scfg = imu_cfg.get("scale", {})
                    if scfg:
                        if "min" in scfg and "max" in scfg:
                            s = float(
                                np.random.uniform(
                                    float(scfg.get("min")), float(scfg.get("max"))
                                )
                            )
                        else:
                            s = float(scfg.get("scale", 1.0))
                        x = x * s

                    # amplitude warping: time-varying multiplicative curve
                    acfg = imu_cfg.get("amplitude_warp", {})
                    if acfg:
                        knot = int(acfg.get("knot", 4))
                        sigma = float(acfg.get("sigma", 0.2))
                        # generate random scale factors at knot points and interpolate
                        knots = np.random.normal(1.0, sigma, size=knot)
                        xp = np.linspace(0, T - 1, num=knot)
                        x_curve = np.interp(np.arange(T), xp, knots).astype(
                            np.float32
                        )  # (T,)
                        curve_t = (
                            torch.from_numpy(x_curve)
                            .to(device=x.device, dtype=x.dtype)
                            .unsqueeze(1)
                        )  # (T,1)
                        x = x * curve_t

                    # permutation: split into segments and permute them
                    rcfg = imu_cfg.get("permutation", {})
                    if rcfg:
                        segments = int(rcfg.get("segments", 4))
                        if segments > 1 and T >= segments:
                            # compute split indices
                            sizes = [T // segments] * segments
                            for i in range(T % segments):
                                sizes[i] += 1
                            parts = []
                            st = 0
                            for sz in sizes:
                                parts.append(x[st : st + sz])
                                st += sz
                            perm_idx = list(range(segments))
                            np.random.shuffle(perm_idx)
                            x = torch.cat([parts[i] for i in perm_idx], dim=0)
                            # if length changed (shouldn't), resample to T
                            if x.shape[0] != T:
                                x = (
                                    F.interpolate(
                                        x.T.unsqueeze(0),
                                        size=T,
                                        mode="linear",
                                        align_corners=False,
                                    )
                                    .squeeze(0)
                                    .T
                                )

                    # time warping: stretch/compress a random segment and resample back to original length
                    twcfg = imu_cfg.get("time_warp", {})
                    if twcfg:
                        seg_min = int(max(2, T // 20))
                        seg_max = int(max(seg_min + 1, T // 3))
                        seg_len = int(np.random.randint(seg_min, seg_max + 1))
                        start = int(np.random.randint(0, max(1, T - seg_len) + 1))
                        factor = float(twcfg.get("factor", np.random.uniform(0.5, 1.5)))
                        end = start + seg_len
                        segment = x[start:end]  # (seg_len, C)
                        new_len = max(2, int(round(segment.shape[0] * factor)))
                        # resample segment in time
                        seg_rs = (
                            F.interpolate(
                                segment.T.unsqueeze(0),
                                size=new_len,
                                mode="linear",
                                align_corners=False,
                            )
                            .squeeze(0)
                            .T
                        )
                        x = torch.cat([x[:start], seg_rs, x[end:]], dim=0)
                        # ensure returned to original length
                        if x.shape[0] != T:
                            x = (
                                F.interpolate(
                                    x.T.unsqueeze(0),
                                    size=T,
                                    mode="linear",
                                    align_corners=False,
                                )
                                .squeeze(0)
                                .T
                            )

                    # rotation (keep existing behavior) - must be last to rotate 3-channel triplets
                    if imu_cfg.get("rotation"):
                        x = apply_rotation_to_sample(x, imu_cfg.get("rotation"))

                    # ensure shape (T, Cc)
                    if x.shape[0] != T:
                        x = (
                            F.interpolate(
                                x.T.unsqueeze(0),
                                size=T,
                                mode="linear",
                                align_corners=False,
                            )
                            .squeeze(0)
                            .T
                        )

                    aug_imus[idx] = x

            if kp_cfg and do_kp_aug.any():
                kp_indices = torch.where(do_kp_aug)[0]
                if len(kp_indices) > 0:
                    # 处理需要KP增强的样本
                    aug_kps_batch = aug_kps_raw[kp_indices].clone()
                    n_kp = len(kp_indices)
                    # possible shapes: (n, T, VC) or (n, T, VC, Vp)
                    print(aug_kps_batch.shape)
                    if aug_kps_batch.ndim == 4:
                        N_kp, T_kp, VC, Vp = aug_kps_batch.shape
                    else:
                        N_kp, T_kp, VC = aug_kps_batch.shape
                        Vp = None

                    # ensure VC is even so we can interpret as (V,2)
                    if VC % 2 == 0:
                        V = VC // 2
                        if Vp is not None:
                            aug_kps_rs = aug_kps_batch.view(n_kp, T_kp, V, 2, Vp)
                        else:
                            aug_kps_rs = aug_kps_batch.view(n_kp, T_kp, V, 2)

                        # scale
                        scfg = kp_cfg.get("scale", {})
                        if scfg:
                            if "min" in scfg and "max" in scfg:
                                scales = np.random.uniform(
                                    float(scfg.get("min")),
                                    float(scfg.get("max")),
                                    size=n_kp,
                                ).astype(np.float32)
                            else:
                                scales = np.full(
                                    (n_kp,),
                                    float(scfg.get("scale", 1.0)),
                                    dtype=np.float32,
                                )
                            scales_t = (
                                torch.from_numpy(scales)
                                .to(
                                    device=aug_kps_batch.device,
                                    dtype=aug_kps_batch.dtype,
                                )
                                .reshape(n_kp, 1, 1, 1, 1 if Vp is not None else 1)
                            )
                            centers = aug_kps_rs.mean(dim=2, keepdim=True)
                            aug_kps_rs = (aug_kps_rs - centers) * scales_t + centers

                        # translation
                        tcfg = kp_cfg.get("translation", {})
                        if tcfg:
                            if "max" in tcfg:
                                maxv = float(tcfg.get("max", 0.0))
                                tx = np.random.uniform(-maxv, maxv, size=n_kp).astype(
                                    np.float32
                                )
                                ty = np.random.uniform(-maxv, maxv, size=n_kp).astype(
                                    np.float32
                                )
                            else:
                                tx = np.full(
                                    (n_kp,),
                                    float(tcfg.get("tx", 0.0)),
                                    dtype=np.float32,
                                )
                                ty = np.full(
                                    (n_kp,),
                                    float(tcfg.get("ty", 0.0)),
                                    dtype=np.float32,
                                )
                            tvec = (
                                torch.from_numpy(np.stack([tx, ty], axis=1))
                                .to(
                                    device=aug_kps_batch.device,
                                    dtype=aug_kps_batch.dtype,
                                )
                                .reshape(n_kp, 1, 1, 2, 1 if Vp is not None else 1)
                            )
                            aug_kps_rs = aug_kps_rs + tvec

                        if Vp is not None:
                            aug_kps_batch = aug_kps_rs.view(n_kp, T_kp, VC, Vp)
                        else:
                            aug_kps_batch = aug_kps_rs.view(n_kp, T_kp, VC)
                        aug_kps_raw[kp_indices] = aug_kps_batch
        if augment_only:
            merged_imus = aug_imus
            merged_kps_raw = aug_kps_raw
            merged_labels = aug_labels
        else:
            merged_imus = torch.cat([merged_imus, aug_imus], dim=0)
            merged_kps_raw = torch.cat([merged_kps_raw, aug_kps_raw], dim=0)
            merged_labels = torch.cat([merged_labels, aug_labels], dim=0)
    # stgcn input compatibility
    # kp tensor may come in multiple shapes:
    #   3 dims: (N, T, VC) where VC = V*C
    #   4 dims: either (N, T, VC, views) or already (N, T, V, C)
    #   5 dims: (N, T, V, C, views)
    if merged_kps_raw.ndim == 3:
        N, T, _ = merged_kps_raw.shape
        V, _, C = 17, 1, 2
        merged_kps = merged_kps_raw.reshape(N, T, V, C)
        merged_kps = merged_kps.permute(0, 3, 1, 2)
    elif merged_kps_raw.ndim == 4:
        # try to guess whether the last dim is "views" or channel
        N, T, D3, D4 = merged_kps_raw.shape
        if D4 == 2:
            # probably already (N,T,V,C)
            merged_kps = merged_kps_raw.permute(0, 3, 1, 2)
        else:
            # treat as (N,T,VC,views): reshape and keep views
            VC = D3
            V = VC // 2
            merged_kps = merged_kps_raw.view(N, T, V, 2, D4)
            # permute to [N, C, T, V, views]
            merged_kps = merged_kps.permute(0, 3, 1, 2, 4)
    elif merged_kps_raw.ndim == 5:
        # assume shape (N, T, V, C, views)
        merged_kps = merged_kps_raw.permute(0, 3, 1, 2, 4)
    else:
        raise ValueError(f"Unsupported kp tensor with ndim={merged_kps_raw.ndim}")
    # merged_kps = merged_kps.unsqueeze(-1)

    indices = torch.randperm(merged_imus.size(0), device=merged_imus.device)

    return merged_imus[indices], merged_kps[indices], merged_labels[indices]


def load_and_split_data(
    data_path: Path,
    user_ids=list(range(1, 16)) + list(range(17, 28)),
    train_size=19,
    val_size=3,
    test_size=4,
    imu_channel: int = 9,
    augment: dict = {},
    logger: logging.Logger | None = config.logger,
    save_generated_dataset: bool = False,
    load_dataset_path: Path | None = None,
):
    """ """

    if load_dataset_path is not None and load_dataset_path.exists():
        if logger:
            logger.info(f"Loading dataset from {load_dataset_path}")
        return torch.load(load_dataset_path)

    random.shuffle(user_ids)

    train_users = user_ids[:train_size]
    val_users = user_ids[train_size : train_size + val_size]
    test_users = user_ids[train_size + val_size : train_size + val_size + test_size]
    # train_users = [1, 3, 4, 6, 8, 9, 12, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
    # val_users = [2, 10, 13]
    # test_users = [7, 5, 11, 14]
    if logger:
        logger.info(f"Train users: {sorted(train_users)}")
        logger.info(f"Val users: {sorted(val_users)}")
        logger.info(f"Test users: {sorted(test_users)}")

    def load_user_data(user_id: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        imu, kps, labels = torch.load(data_path / f"user_{user_id}.pt")
        return imu[:, :, :imu_channel], kps, labels

    train_data = [load_user_data(uid) for uid in train_users]
    val_data = [load_user_data(uid) for uid in val_users]
    test_data = [load_user_data(uid) for uid in test_users]

    train_set = merge_data(train_data, apply_augmentation=True, augment=augment)
    val_set = merge_data(val_data, apply_augmentation=False)
    test_set = merge_data(test_data, apply_augmentation=False)
    if logger:
        logger.info(f"Train set size: {len(train_set[0])}")
        logger.info(f"Validation set size: {len(val_set[0])}")
        logger.info(f"Test set size: {len(test_set[0])}")

    if save_generated_dataset:
        torch.save((train_set, val_set, test_set), data_path / f"new_dataset.pt")
        if logger:
            logger.info(f"Saved generated dataset to {data_path / f'new_dataset.pt'}")

    return train_set, val_set, test_set


def augment_saved_dataset(dataset, augment: dict):
    imu, kp, label = dataset
    imu, kp, label = imu.clone(), kp.clone(), label.clone()
    if kp[0].ndim == 3:
        dataset = [
            (
                imu[i].unsqueeze(0),
                kp[i].permute(1, 2, 0).flatten(start_dim=1).unsqueeze(0),
                label[i].unsqueeze(0),
            )
            for i in range(imu.shape[0])
        ]
    else:
        dataset = [
            (
                imu[i].unsqueeze(0),
                kp[i].permute(1, 2, 0, 3).flatten(start_dim=1, end_dim=2).unsqueeze(0),
                label[i].unsqueeze(0),
            )
            for i in range(imu.shape[0])
        ]
    return merge_data(dataset, apply_augmentation=True, augment=augment)


def add_single_modal_samples(dataset, rate=0.01):
    """
    在训练集中添加单模态样本
    :param trainset: 原始数据集，格式为[(imu_tensor, kp_tensor, label), ...]
    :return: 增强后的数据集
    """
    # 1. 按类别分组原始样本
    class_samples = defaultdict(list)
    for imu, kp, label in zip(*dataset):
        # 确保label是整数（处理tensor类型）
        label_val = label.item() if isinstance(label, torch.Tensor) else int(label)
        class_samples[label_val].append((imu, kp, label))

    # 2. 计算原始数据集大小
    N = len(dataset[2])

    # 3. 计算每个类别需要添加的单模态样本数
    # 理论值: m = N / 180 (每个类别每种模态)
    # 取整确保整数且比例均衡
    m_per_class = round(N * rate / 10 / (1 - 2 * rate))
    new_samples = {
        "imu": [],
        "kp": [],
        "label": [],
    }
    # 4. 生成单模态样本
    for class_id in range(10):  # 遍历0-9所有类别
        if class_id not in class_samples:
            continue

        samples = class_samples[class_id]
        # 随机选择样本（可重复采样确保数量）
        selected = random.choices(samples, k=m_per_class)

        for imu, kp, label in selected:
            # 创建IMU单模态样本 (KP置零)
            zero_kp = torch.zeros_like(kp)
            new_samples["imu"].append(imu.clone())
            new_samples["kp"].append(zero_kp)
            new_samples["label"].append(label)

            # 创建KP单模态样本 (IMU置零)
            zero_imu = torch.zeros_like(imu)
            new_samples["imu"].append(zero_imu)
            new_samples["kp"].append(kp.clone())
            new_samples["label"].append(label)
    # 5. 合并数据集
    enhanced_dataset = (
        torch.cat([dataset[0], torch.stack(new_samples["imu"], dim=0)], dim=0),
        torch.cat([dataset[1], torch.stack(new_samples["kp"], dim=0)], dim=0),
        torch.cat([dataset[2], torch.stack(new_samples["label"], dim=0)], dim=0),
    )

    return enhanced_dataset


class MultiModalTensorDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        imu_tensor: torch.Tensor,
        kp_tensor: torch.Tensor,
        labels: torch.Tensor,
    ):
        assert len(imu_tensor) == len(kp_tensor) == len(labels)
        self.imu = imu_tensor
        self.kp = kp_tensor
        self.labels = labels

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, idx):
        return {"imu": self.imu[idx], "kp": self.kp[idx]}, self.labels[idx]


def get_dataloaders(
    train_set,
    val_set,
    test_set,
    batch_size=32,
    seed=3407,
    data_type: Literal["imu", "kp", "both"] = "imu",
    logger: logging.Logger = config.logger,
):
    X_imu_train, X_kp_train, y_train = train_set
    X_imu_val, X_kp_val, y_val = val_set
    X_imu_test, X_kp_test, y_test = test_set

    match data_type:
        case "imu":
            train_loader = DataLoader(
                TensorDataset(X_imu_train, y_train),
                batch_size=batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(seed),
            )
            val_loader = DataLoader(
                TensorDataset(X_imu_val, y_val), batch_size=batch_size, shuffle=False
            )
            test_loader = DataLoader(
                TensorDataset(X_imu_test, y_test), batch_size=batch_size, shuffle=False
            )
        case "kp":
            train_loader = DataLoader(
                TensorDataset(X_kp_train, y_train),
                batch_size=batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(seed),
            )
            val_loader = DataLoader(
                TensorDataset(X_kp_val, y_val), batch_size=batch_size, shuffle=False
            )
            test_loader = DataLoader(
                TensorDataset(X_kp_test, y_test), batch_size=batch_size, shuffle=False
            )
        case "both":
            train_ds = MultiModalTensorDataset(X_imu_train, X_kp_train, y_train)
            val_ds = MultiModalTensorDataset(X_imu_val, X_kp_val, y_val)
            test_ds = MultiModalTensorDataset(X_imu_test, X_kp_test, y_test)

            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(seed),
            )
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        case _:
            raise NotImplementedError(f"Invalid data type {data_type}")
    if logger:
        logger.info(f"Using data type {data_type}")

    return train_loader, val_loader, test_loader
