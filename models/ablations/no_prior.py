"""Direct RAMoE ablation definitions, preserving the original class bodies."""
import torch
import torch.nn as nn
import torch.nn.functional as F


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
