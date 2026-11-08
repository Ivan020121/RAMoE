import torch
import torch.nn as nn
import torch.nn.functional as F


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
        num_classes=10,
        history=None,
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

        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim * 4),
            nn.Linear(fusion_dim * 4, fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )
        self.history = {
            "imu": [0.720, 0.739, 0.730, 0.603, 0.782, 0.630, 0.793, 0.718, 0.898, 0.805],
            "kp": [0.682, 0.585, 0.762, 0.691, 0.945, 0.947, 0.962, 0.895, 0.933, 0.791],
        }
        if history:
            self.history = history

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

        s = self.classifier(torch.concat([modal_out, moe_output], dim=1)) # (B, 4*fusion_dim) -> # (B, 10)
        return s
