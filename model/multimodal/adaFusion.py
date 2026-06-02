import torch
import torch.nn as nn
import torch.nn.functional as F

from model.multimodal.bili import GatedAttentionLayer

from mamba_ssm import Mamba2

class CrossModalAttentionFusion(nn.Module):
    def __init__(self, imu_dim=1024, kp_dim=256, hidden_dim=1024, num_heads=8, dropout=0.3):
        super().__init__()
        
        # 投影层对齐维度
        self.imu_proj = nn.Linear(imu_dim, hidden_dim)
        
        self.kp_proj = nn.Linear(kp_dim, hidden_dim)
        
        self.modal_gate = nn.Linear(128*2, 2)

        nn.init.xavier_normal_(self.modal_gate.weight)
        nn.init.zeros_(self.modal_gate.bias)

        # 多头注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.num_heads = num_heads
        self.channel = hidden_dim//num_heads
        self.imu_proj2fusion = nn.Linear(hidden_dim, self.channel)

        # self.cross_attention = GatedAttentionLayer(
        #     embed_dim=hidden_dim,
        #     num_heads=num_heads,
        #     dropout=dropout,
        #     pre_norm=True,
        #     use_flash_attention=False,
        #     is_cross_attention=True
        # )
        

        self.adaptive_dynamic_heads_cma = AdaptiveDynamicHeadsCMA(self.num_heads, self.channel)

        # Residual connection 
        self.residual_alpha = nn.Parameter(torch.ones(self.channel) * 0.5)

        # Transformer decoder
        self.decoder_layer = nn.TransformerDecoderLayer(d_model = self.channel, nhead = 8, batch_first=True, norm_first = True, dropout = dropout)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers = 4)
        self.mamba = Mamba2(self.channel ,d_state= 32,d_conv= 4)
        # 分类头
        self.classifier = nn.Sequential(
            # nn.LayerNorm(self.channel*2),
            nn.Linear(self.channel, 10),
        )
    
    def forward(self, imu_features, kp_features):
        """
        imu_features: (B, 13, 1024)
        kp_features: (B, 256, 13, 17)
        """
        B, T_imu, C_imu = imu_features.shape
        B, C_kp, T_kp, V_kp = kp_features.shape
        assert T_imu == T_kp, f"时间维度不一致: IMU={T_imu}, KP={T_kp}"
        
        kp_pooled = F.adaptive_avg_pool2d(kp_features, (T_kp, 1))  # (B, 256, 13, 1)
        kp_pooled = kp_pooled.squeeze(-1).permute(0, 2, 1)  # (B, 13, 256)  
        
        imu_proj = self.imu_proj(imu_features)  # (B, 13, hidden_dim)
        kp_proj = self.kp_proj(kp_pooled)  # (B, 13, hidden_dim)

        # imu_mean = F.adaptive_avg_pool2d(imu_proj, (1,128)).squeeze(1)
        # kp_mean = F.adaptive_avg_pool2d(kp_proj, (1,128)).squeeze(1)
        # modal_gate = F.softmax(self.modal_gate(torch.concat([imu_mean, kp_mean], dim=1)), dim=1)
        # modal_fusion = imu_mean * modal_gate[:,0].unsqueeze(1) + kp_mean * modal_gate[:,1].unsqueeze(1)

        attn_output, _  = self.cross_attention(imu_proj, kp_proj, kp_proj)

        fused = self.adaptive_dynamic_heads_cma(attn_output)                # [B, C, N]

        #------ Residual Fusion 
        alpha = self.residual_alpha.view(1, 1, self.channel)
        imu_proj = self.imu_proj2fusion(imu_proj)      
        cross_out = alpha * fused + (1 - alpha) * imu_proj                   # [B, N, C]
        
        # dec_out = self.decoder(cross_out, cross_out)   
        #dec_out = self.mamba(cross_out)                       # [B, N, C]

        temporal_pooled = torch.mean(cross_out, dim=1)  # (B, N)
        output = self.classifier(temporal_pooled)
        
        return output

class AdaptiveDynamicHeadsCMA(nn.Module):
    """
    Small network to compute per-head weights from the concatenated heads
    Input shape: [B, N, H*C]      Output shape: [B, N, H]
    """
    def __init__(self, num_heads, channel):
        super().__init__()
        self.num_heads = num_heads
        self.channel = channel
        
        self.gate_mlp = nn.Sequential(
            nn.Linear(num_heads * channel, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_heads)
        )

    def forward(self, combined):
  
        B, N, C = combined.shape                                        # [B, N, H*C]
        H = self.num_heads


        gates = self.gate_mlp(combined)                                 # raw scores: [B, N, H]
        gates = F.softmax(gates, dim=-1)                                # [B, N, H]
        combined = combined.permute(0, 2, 1).contiguous()
        combined_heads = combined.view(B, H, C//H, N).permute(0, 1, 3, 2)  # [B, H, N, C]
        gates = gates.permute(0, 2, 1).unsqueeze(-1)                    # [B, H, N, 1]
        
        weighted_heads = combined_heads * gates                         # [B, H, C, N] * [B, H, N, 1] → broadcasting
        weighted_heads = weighted_heads.permute(0, 1, 3, 2)             # back to [B, H, C, N]
        
        fused = weighted_heads.sum(dim=1)                               # [B, C, N]
        fused = fused.permute(0, 2, 1)                                  # [B, N, C]
        return fused

class FusionAttn(nn.Module):
    def __init__(self, imu_encoder, kp_encoder, fusion_dim=512, imu_embedding_dim=1024, kp_embedding_dim=256):
        super().__init__()
        self.imu_encoder = imu_encoder
        self.kp_encoder = kp_encoder

        for param in self.imu_encoder.parameters():
            param.requires_grad = False
        for param in self.kp_encoder.parameters():
            param.requires_grad = False

        self._modify_encoders()
        
        self.fusion = CrossModalAttentionFusion(
            imu_dim=imu_embedding_dim,
            kp_dim=kp_embedding_dim,
            hidden_dim=fusion_dim,
            num_heads= 8,
            dropout=0.1
        )
        self.apply_moddrop=False
        
    def _modify_encoders(self):
        """修改编码器以返回中间特征而不是池化后的特征"""
        # 修改IMU编码器：返回GRU输出
        
        def new_imu_forward(x):
            # 复制原始forward中的处理逻辑
            x = torch.permute(x, (0, 2, 1))
            x = self.imu_encoder.net(x)
            x = torch.permute(x, (0, 2, 1))
            output, h_n = self.imu_encoder.rnn(x)
            return output  # 返回所有时间步的输出 (B, 13, 1024)
        
        self.imu_encoder.forward = new_imu_forward
        
        # 修改KP编码器：返回最后一个卷积层的输出
        
        def new_kp_forward(x):
            # 复制原始forward中的处理逻辑到池化之前
            N, C, T, V = x.size()
            x = x.permute(0, 3, 1, 2).contiguous()
            x = x.view(N, V * C, T)
            x = self.kp_encoder.data_bn(x)
            x = x.view(N, V, C, T)
            x = x.permute(0, 2, 3, 1).contiguous()
            x = x.view(N, C, T, V)
            
            for gcn, importance in zip(self.kp_encoder.st_gcn_networks, self.kp_encoder.edge_importance):
                x, _ = gcn(x, self.kp_encoder.A * importance)
            
            # 不进行全局池化，返回特征图
            return x  # (B, 256, T', 17)
        
        self.kp_encoder.forward = new_kp_forward
    
    def _apply_moddrop(self, imu_features: torch.Tensor, kp_features: torch.Tensor):
        prob = 0.333
        modal_mask = torch.bernoulli(torch.tensor([prob, prob]))

        
        if not torch.all(modal_mask < prob):
            if modal_mask[0].item() < prob:
                imu_features = torch.zeros_like(imu_features)
            
            if modal_mask[1].item() < prob:
                kp_features = torch.zeros_like(kp_features)
        
        return imu_features, kp_features

    def forward(self, imu_input, kp_input):
        # 获取中间特征
        if self.apply_moddrop:
            imu_input, kp_input = self._apply_moddrop(imu_input, kp_input)
        imu_features = self.imu_encoder(imu_input)  # (B, 134, 1024)
        kp_features = self.kp_encoder(kp_input)     # (B, 256, 13, 17)
        imu_features = torch.concat([imu_features[:,:4,:], imu_features[:, 5:, :]], dim=1)
        
        # 融合
        output = self.fusion(imu_features, kp_features)
        
        return output