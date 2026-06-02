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

    # def _build_expert(self, in_dim, out_dim):
    #     return nn.Sequential(
    #         nn.Conv1d(in_dim, out_dim // 2, kernel_size=3, padding=1),
    #         nn.BatchNorm1d(out_dim // 2),
    #         nn.ReLU(True),
    #         nn.Conv1d(out_dim // 2, out_dim, kernel_size=3, padding=1),
    #         nn.BatchNorm1d(out_dim),
    #     )
    def _build_expert(self, in_dim, out_dim):
        # 建议：将 num_groups 设为 8 或 4，确保能被 out_dim//2 和 out_dim 整除
        # 如果 out_dim 较小，可设为 num_groups=1 (即 LayerNorm 效果) 或 out_dim (即 InstanceNorm 效果)
        gn_hidden = min(2, out_dim // 2) 
        gn_output = min(2, out_dim)
        
        return nn.Sequential(
            nn.Conv1d(in_dim, out_dim // 2, kernel_size=3, padding=1),
            # 替换 BatchNorm1d -> GroupNorm (num_groups, num_channels)
            nn.GroupNorm(num_groups=gn_hidden, num_channels=out_dim // 2),
            nn.ReLU(True),
            nn.Conv1d(out_dim // 2, out_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=gn_output, num_channels=out_dim),
        )

    def normalize_scores(self, imu_score, kp_score):
        imu_score = (imu_score - imu_score.mean(dim=1, keepdim=True)) / (imu_score.std(dim=1, keepdim=True) + 1e-8)
        kp_score = (kp_score - kp_score.mean(dim=1, keepdim=True)) / (kp_score.std(dim=1, keepdim=True) + 1e-8)
        return imu_score, kp_score

    def forward(self, imu_input: torch.Tensor, kp_input: torch.Tensor):
        imu_features, imu_score = self.imu_encoder(imu_input)  # (B, fusion_dim)
        kp_features, kp_score = self.kp_encoder(kp_input)      # (B, fusion_dim)
        imu_score, kp_score = self.normalize_scores(imu_score, kp_score)

        modal_mask = self.modality_selector(imu_score, kp_score) # (B, 2)
        imu_expert_out = self.imu_expert(imu_features.unsqueeze(-1)).squeeze(-1)
        kp_expert_out = self.kp_expert(kp_features.unsqueeze(-1)).squeeze(-1)

        modal_out = torch.stack([imu_expert_out, kp_expert_out], dim=1) * modal_mask.unsqueeze(-1)
        modal_out = modal_out.flatten(1) # (B, 2*fusion_dim)

        feat = torch.cat([imu_features, kp_features], dim=1) # (B, 2*fusion_dim)

        logits = self.router(feat)
        noise_logits = self.noise_linear(feat)
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1) # indices: (B, top_k)
        
        zeros = torch.full_like(noisy_logits, float("-inf"))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_weights = F.softmax(sparse_logits, dim=1)  # (B, expert_num)

        moe_output = torch.zeros_like(feat) 
        
        for k in range(self.top_k):
            current_expert_indices = indices[:, k]  # (B,)
            current_weights = router_weights[:, k]  # (B,)
            
            # 避免重复加载专家
            unique_experts = torch.unique(current_expert_indices)
            
            for exp_id in unique_experts:
                sample_mask = (current_expert_indices == exp_id)
                if not sample_mask.any():
                    continue
                
                selected_feat = feat[sample_mask].unsqueeze(-1)
                expert_out = self.experts[exp_id](selected_feat).squeeze(-1)
                
                weighted_out = expert_out * current_weights[sample_mask].unsqueeze(-1)
                moe_output[sample_mask] += weighted_out

        s = self.classifier(torch.concat([modal_out, moe_output], dim=1))
        return s
