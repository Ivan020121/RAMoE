from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from multimodal.train.base import get_unimodal_models
from util.torch_utils import (
    add_single_modal_samples,
    fix_random_seed,
    get_dataloaders,
    load_and_split_data,
)
from config import config


class ModalitySelector:
    def __init__(self, imu_class_f1, kp_class_f1, device):
        self.w_imu = torch.tensor(imu_class_f1, dtype=torch.float32, device=device)
        self.w_kp = torch.tensor(kp_class_f1, dtype=torch.float32, device=device)

    def __call__(self, imu_logits, kp_logits):
        p_imu = F.softmax(imu_logits, dim=1)
        p_kp = F.softmax(kp_logits, dim=1)

        # 2. 获取模型“当前最想选的类别”以及“它对该类别的自信程度”
        max_prob_imu, pred_idx_imu = torch.max(p_imu, dim=1)
        max_prob_kp, pred_idx_kp = torch.max(p_kp, dim=1)

        # 3. 获取历史可靠性 (查表)
        rel_imu = self.w_imu[pred_idx_imu]
        rel_kp = self.w_kp[pred_idx_kp]

        # 4. 计算综合信任分 (Trust Score)
        # Score = 当前自信度 * 历史靠谱程度
        score_imu = max_prob_imu * rel_imu
        score_kp = max_prob_kp * rel_kp

        return F.softmax(torch.stack([score_imu, score_kp], dim=1), dim=1)


class PRISM(nn.Module):
    def __init__(
        self,
        imu_encoder,
        kp_encoder,
        fusion_dim=512,
        dropout=0.1,
        expert_num=4,
        topk=1,
        device="cuda",
    ):
        super().__init__()
        self.imu_encoder = imu_encoder
        self.kp_encoder = kp_encoder

        for param in self.imu_encoder.parameters():
            param.requires_grad = False
        for param in self.kp_encoder.parameters():
            param.requires_grad = False

        self._modify_encoders()

        self.expert_num = expert_num
        self.imu_expert = self._build_expert(fusion_dim, fusion_dim)
        self.kp_expert = self._build_expert(fusion_dim, fusion_dim)
        self.top_k = topk
        self.router = nn.Sequential(
            nn.Linear(fusion_dim * 2, expert_num),
        )

        self.classifier1 = nn.Sequential(
            nn.LayerNorm(fusion_dim * 2),
            nn.Linear(fusion_dim * 2, fusion_dim),
        )
        self.classifier2 = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 10)
        )
        self.history = {
            "imu": [0.720, 0.739, 0.730, 0.603, 0.782, 0.630, 0.793, 0.718, 0.898, 0.805],
            "kp": [0.682, 0.585, 0.762, 0.691, 0.945, 0.947, 0.962, 0.895, 0.933, 0.791],
        }

        self.modality_selector = ModalitySelector(
            self.history["imu"], self.history["kp"], device
        )

    def _modify_encoders(self):
        # 冻结/解冻特定层参数
        for name, param in self.imu_encoder.encoder.named_parameters():
            param.requires_grad = "rnn" in name

        total_layers = len(self.kp_encoder.encoder.st_gcn_networks)
        for i, gcn in enumerate(self.kp_encoder.encoder.st_gcn_networks):
            for param in gcn.parameters():
                param.requires_grad = i >= total_layers - 2

        ...

    def _build_expert(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Conv1d(in_dim, out_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_dim // 2),
            nn.ReLU(True),
            nn.Conv1d(out_dim // 2, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_dim),
        )

    def forward(self, imu_input: torch.Tensor, kp_input: torch.Tensor):
        # imu_input: (B, T_imu_raw, C_imu), kp_input: (B, C_kp, T_kp, V)
        imu_features, imu_score = self.imu_encoder(imu_input)  # (B, fusion_dim)
        kp_features, kp_score = self.kp_encoder(kp_input)  # (B, fusion_dim)

        # modal expert
        modal_mask = self.modality_selector(imu_score, kp_score) # (B, 2)
        imu_expert_out = self.imu_expert(imu_features.unsqueeze(-1)).squeeze(-1) # (B, fusion_dim)
        kp_expert_out = self.kp_expert(kp_features.unsqueeze(-1)).squeeze(-1) # (B, fusion_dim)

        modal_out = torch.stack(
            [imu_expert_out, kp_expert_out], dim=1
        ) * modal_mask.unsqueeze(-1) # (B, 2, fusion_dim)
        modal_out = modal_out.flatten(1) # (B, 2*fusion_dim)

        mid_s = self.classifier1(modal_out) # (B, 2*fusion_dim) -> # (B, 10)
        s = self.classifier2(mid_s)
        return s, mid_s

def pca2D(model, test_loader, device, pic_tag=""):
    model.to(device)
    model.eval()

    all_preds = []
    all_mid_feats = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x = {k: v.to(device).float() for k, v in x.items()}
            y = y.to(device).long()
            output = model(x["imu"], x["kp"])

            all_mid_feats.append(output[1])
            output = output[0]

            _, pred = output.max(1)
            all_preds.append(pred.cpu())
            all_labels.append(y.cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    mid_feats = torch.cat(all_mid_feats).cpu().numpy()  # shape: (n_samples, 512)

    # 1. 特征标准化
    scaler = StandardScaler()
    feats_scaled = scaler.fit_transform(mid_feats)

    # 2. PCA降维
    pca_obj = PCA(n_components=2)
    feats_reduced = pca_obj.fit_transform(feats_scaled)

    # 3. 可视化（仅预测标签 + 完整类别图例）
    cmap = plt.colormaps.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(
        feats_reduced[:, 0],
        feats_reduced[:, 1],
        c=y_pred,
        cmap=cmap,
        alpha=0.7,
        s=30,
        edgecolors="w",
        linewidths=0.5,
    )
    ax.tick_params(axis='both', labelsize=16)
    # ax.set_xlabel(f"PC1 ({pca_obj.explained_variance_ratio_[0]:.2%})", fontsize=16)
    # ax.set_ylabel(f"PC2 ({pca_obj.explained_variance_ratio_[1]:.2%})", fontsize=16)
    ax.set_title("")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.text(0.5, -0.12, f"{pic_tag}", transform=ax.transAxes, 
            ha='center', va='top', fontsize=26, fontweight='bold')

    plt.tight_layout()
    plt.savefig(
        f"pca_2D_{pic_tag}.svg",
        dpi=300,
        format="svg",
        bbox_inches="tight",
    )
    plt.show()

    # 4. 打印解释方差
    print(f"Explained variance ratio: {pca_obj.explained_variance_ratio_}")
    print(f"Total variance explained: {pca_obj.explained_variance_ratio_.sum():.2%}")

    return feats_reduced, y_true, y_pred, pca_obj


if __name__ == "__main__":
    logger = config.logger
    seed = 3407
    imu_channels = 9
    train_epoch = 40
    repeat_train = 1
    checkpoint_path = Path("checkpoint/multimodal")
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
    fusion_dim = 512

    imu_encoder, kp_encoder = get_unimodal_models(
        train_set,
        val_set,
        test_set,
        device=device,
        no_train=True,
        with_head=True,
        return_both=True,
    )

    train_loader, val_loader, test_loader = get_dataloaders(
        add_single_modal_samples(train_set),
        val_set,
        test_set,
        batch_size=batch_size,
        data_type="both",
    )
    model = PRISM(imu_encoder, kp_encoder, fusion_dim, dropout=0.3, device=device)

    model.load_state_dict(
        torch.load(checkpoint_path / "PRISM 2026-02-01 23:10:13-latest.pt"), strict=False
    )
    pca2D(model, test_loader, device, pic_tag="No IMU")
