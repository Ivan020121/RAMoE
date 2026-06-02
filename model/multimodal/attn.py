import torch
import torch.nn as nn
import torch.nn.functional as F

from model.multimodal.bili import GatedAttentionLayer

class CrossModalAttentionFusion(nn.Module):
    def __init__(self, imu_dim=512, kp_dim=256, hidden_dim=512, num_heads=8, dropout=0.3):
        super().__init__()
        
        # 投影层对齐维度
        self.imu_proj = nn.Linear(imu_dim, hidden_dim)
        self.kp_proj = nn.Linear(kp_dim, hidden_dim)
        
        # 多头注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

    
    def forward(self, imu_features, kp_features):
        """
        imu_features: (B, 13, 1024)
        kp_features: (B, 256, 13, 17)
        """        
        imu_proj = self.imu_proj(imu_features)  # (B, hidden_dim)
        kp_proj = self.kp_proj(kp_features)  # (B, hidden_dim)

        attn_output, _  = self.cross_attention(imu_proj, kp_proj, kp_proj)
        
        fused = torch.cat([imu_proj, attn_output, kp_proj], dim=-1)  # (B, 13, hidden_dim*3)
                
        return fused

class MultiModalAttn(nn.Module):
    def __init__(self, imu_encoder, kp_encoder, fusion_dim=512, imu_embedding_dim=512, kp_embedding_dim=256, dropout=0.1):
        super().__init__()
        self.imu_encoder = imu_encoder
        self.kp_encoder = kp_encoder

        for param in self.imu_encoder.parameters():
            param.requires_grad = False
        for param in self.kp_encoder.parameters():
            param.requires_grad = False

        
        self.fusion = CrossModalAttentionFusion(
            imu_dim=imu_embedding_dim,
            kp_dim=kp_embedding_dim,
            hidden_dim=fusion_dim,
            num_heads= 8,
            dropout=dropout
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim*3),
            nn.Linear(fusion_dim*3, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 10),
        )

        self.apply_moddrop=False

    
    def _apply_moddrop(self, imu_features: torch.Tensor, kp_features: torch.Tensor):
        modal_mask = torch.bernoulli(torch.tensor([0.5, 0.5]))

        
        if not torch.all(modal_mask < 0.5):
            if modal_mask[0].item() < 0.5:
                imu_features = torch.zeros_like(imu_features)
            
            if modal_mask[1].item() < 0.5:
                kp_features = torch.zeros_like(kp_features)
        
        return imu_features, kp_features

    def forward(self, imu_input, kp_input):
        # 获取中间特征
        if self.apply_moddrop:
            imu_input, kp_input = self._apply_moddrop(imu_input, kp_input)
        imu_features = self.imu_encoder(imu_input)  # (B, 134, 1024)
        kp_features = self.kp_encoder(kp_input)     # (B, 256, 13, 17)
        # imu_features = torch.concat([imu_features[:,:8,:], imu_features[:, 9:, :]], dim=1)
        
        # 融合
        fused = self.fusion(imu_features, kp_features)

        output = self.classifier(fused)
        return output