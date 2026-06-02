from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from multimodal.evaluate.pcaa import pca2D
from multimodal.train.base import Trainer, get_unimodal_models
from util.torch_utils import (
    add_single_modal_samples,
    fix_random_seed,
    get_dataloaders,
    load_and_split_data,
    multi_evaluate,
)
from config import config


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

        self.noise_linear = nn.Linear(fusion_dim * 2, expert_num)
        self.expert_num = expert_num
        self.experts = nn.ModuleList(
            [
                self._build_expert(fusion_dim * 2, fusion_dim * 2)
                for _ in range(expert_num)
            ]
        )
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
            nn.Linear(fusion_dim, 10),
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

        feat = torch.cat([imu_features, kp_features], dim=1) # (B, 2*fusion_dim)

        # 路由权重计算
        logits = self.router(feat)  # (B, expert_num)
        noise_logits = self.noise_linear(feat) # (B, expert_num)

        # 向 logits 添加缩放的单位高斯噪声
        noise = torch.randn_like(logits) * F.softplus(noise_logits) # (B, expert_num)
        noisy_logits = logits + noise # (B, expert_num)

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float("-inf"))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_weights = F.softmax(sparse_logits, dim=1)  # (B, expert_num)

        expert_outputs = torch.stack(
            [expert(feat.unsqueeze(-1)) for expert in self.experts], dim=1
        ).squeeze(-1) # (B, expert_num, 2*fusion_dim)

        # 加权融合
        moe_output = (router_weights.unsqueeze(-1) * expert_outputs).mean(1) # (B, 2*fusion_dim)

        mid_s = self.classifier1(moe_output) # (B, 2*fusion_dim) -> # (B, 10)
        s = self.classifier2(mid_s)
        return s, mid_s


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

    for i in range(repeat_train):
        imu_encoder, kp_encoder = get_unimodal_models(
            train_set,
            val_set,
            test_set,
            device=device,
            pretrained_names=(
                "unimodal_imu 2026-01-31 17:49:32-latest.pt",
                "unimodal_kp 2026-01-25 13:00:01-latest.pt",
            ),
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
        # trainer = Trainer(
        #     model,
        #     "PRISM",
        #     train_loader,
        #     val_loader,
        #     optimizer=optim.AdamW(model.parameters(), lr=lr),
        #     num_epochs=train_epoch,
        #     test_loader=test_loader,
        #     device=device,
        #     modal=modal,
        #     checkpoint_path=checkpoint_path,
        #     use_scheduler=False,
        #     use_early_stopping=True,
        #     patience=6,
        # )

        # trainer.train()

        # multi_evaluate(
        #     model,
        #     f"{model_name}",
        #     test_loader,
        #     device=device,
        #     modal=modal,
        #     checkpoint_best_path=checkpoint_path
        #     / f"{trainer.model_weight_prefix}-best.pt",
        # )
        model.load_state_dict(
            torch.load(checkpoint_path / "PRISM 2026-02-01 23:02:37-latest.pt"), strict=False
        )
        pca2D(model, test_loader, device, pic_tag="No Prior")
