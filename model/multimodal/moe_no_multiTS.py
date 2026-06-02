import torch
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
    
class CrossModalAttentionFusion(nn.Module):
    def __init__(self, hidden_dim=1024, num_heads=8, dropout=0.3):
        super().__init__()
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
  
    def forward(self, imu_features, kp_features):
        imu_enhance_kp, _  = self.cross_attention(imu_features, kp_features, kp_features)
        kp_enhance_imu, _  = self.cross_attention(kp_features, imu_features, imu_features)
  
        return kp_enhance_imu, imu_enhance_kp

class MoEAttn(nn.Module):
    def __init__(self, imu_encoder, kp_encoder, fusion_dim=512, imu_embedding_dim=512, kp_embedding_dim=256, dropout=0.1):
        super().__init__()
        self.imu_encoder = imu_encoder
        self.kp_encoder = kp_encoder

        for param in self.imu_encoder.parameters():
            param.requires_grad = False
        for param in self.kp_encoder.parameters():
            param.requires_grad = False

        self._modify_encoders()
        
        self.imu_proj = nn.Linear(imu_embedding_dim, fusion_dim)
        self.kp_proj = nn.Linear(kp_embedding_dim, fusion_dim)
        

        self.cross_attn = CrossModalAttentionFusion(
            hidden_dim=fusion_dim,
            num_heads= 16,
            dropout=dropout
        )

        self.experts = nn.ModuleDict({
            'imu': self._build_expert(fusion_dim, fusion_dim),
            'kp': self._build_expert(fusion_dim, fusion_dim),
            'imukp': self._build_expert(fusion_dim, fusion_dim),
            'kpimu': self._build_expert(fusion_dim, fusion_dim),
        })
        self.router_mix = Mlp(fusion_dim, fusion_dim // 2, 10, drop=dropout)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, 10),
        )
        
    def _modify_encoders(self):
        # 冻结/解冻特定层参数
        for name, param in self.imu_encoder.named_parameters():
            param.requires_grad = "rnn" in name
        
        total_layers = len(self.kp_encoder.st_gcn_networks)
        for i, gcn in enumerate(self.kp_encoder.st_gcn_networks):
            for param in gcn.parameters():
                param.requires_grad = (i >= total_layers - 2)
        
    

    def _build_expert(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Conv1d(in_dim, out_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_dim // 2),
            nn.ReLU(True),
            nn.Conv1d(out_dim // 2, out_dim, kernel_size=1),
            nn.BatchNorm1d(out_dim)
        )

    def forward(self, imu_input: torch.Tensor, kp_input: torch.Tensor):
        # imu_input: (B, T_imu_raw, C_imu), kp_input: (B, C_kp, T_kp, V)
        imu_features = self.imu_encoder(imu_input)  # (B, T_imu, D_imu=1024)
        kp_features = self.kp_encoder(kp_input)     # (B, D_kp=256, T_kp, V=17)
        
        
        imu_proj = self.imu_proj(imu_features)  # (B, 133, D_fusion=512)
        kp_proj = self.kp_proj(kp_features)      # (B, 13, 512)
        
        kp_enhance_imu, imu_enhance_kp = self.cross_attn(imu_proj, kp_proj)  # (B,13,512), (B,133,512)

        # 转置为Conv1d所需格式: (B, D, T)
        imu_attn_t = kp_enhance_imu.unsqueeze(-1)       # (B, 512, 133)
        kp_attn_t = imu_enhance_kp.unsqueeze(-1)        # (B, 512, 13)
        imu_raw_t = imu_proj.unsqueeze(-1)              # (B, 512, 133)
        kp_raw_t = kp_proj.unsqueeze(-1)                # (B, 512, 13)

        # 专家网络处理
        out_imu = self.experts['imu'](imu_raw_t).squeeze(-1)        # (B, 133, 512)
        out_kp = self.experts['kp'](kp_raw_t).squeeze(-1)           # (B, 13, 512)
        out_imukp = self.experts['imukp'](kp_attn_t).squeeze(-1)    # (B, 13, 512)
        out_kpimu = self.experts['kpimu'](imu_attn_t).squeeze(-1)   # (B, 133, 512)

        # 路由权重计算
        router_logits = torch.stack([
            self.router_mix(out_imu).mean(-1),      # (B,)
            self.router_mix(out_kp).mean(-1),
            self.router_mix(out_imukp).mean(-1),
            self.router_mix(out_kpimu).mean(-1)
        ], dim=1)  # (B, 4)
        
        router_weights = F.softmax(router_logits, dim=1)  # (B, 4)

        # 专家输出堆叠
        expert_outputs = torch.stack([
            out_imu, out_kp, out_imukp, out_kpimu
        ], dim=1)  # (B, 4, 512)

        # 加权融合
        moe_output = router_weights.unsqueeze(-1) * expert_outputs  # (B, 4, 512)

        s = self.classifier(moe_output).mean(1)  # (B, 4, 10) -> (B, 10)
        return s