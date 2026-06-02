import math
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
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, imu_features, kp_features):
        imu_enhance_kp, _ = self.cross_attention(imu_features, kp_features, kp_features)
        kp_enhance_imu, _ = self.cross_attention(
            kp_features, imu_features, imu_features
        )

        return kp_enhance_imu, imu_enhance_kp


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(RMSNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        mean = (x**2).mean(-1, keepdim=True)
        out_mean = x / torch.sqrt(mean + self.eps)
        out = self.gamma * out_mean
        return out


def sinkhorn_knopp_batched(A, it=1000, eps=1e-8):
    """
    A is not negative matrix
    """
    batch_size, n, _ = A.shape

    u = torch.ones(batch_size, n, device=A.device)
    v = torch.ones(batch_size, n, device=A.device)

    for _ in range(it):
        v_temp = v.unsqueeze(2)  # (B, n, 1)
        Av = torch.bmm(A, v_temp).squeeze(2)  # (B, n)
        u = 1.0 / (Av + eps)

        u_temp = u.unsqueeze(2)  # (B, n, 1)
        At_u = torch.bmm(A.transpose(1, 2), u_temp).squeeze(2)
        v = 1.0 / (At_u + eps)

    U = torch.diag_embed(u)  # (B, n, n)
    V = torch.diag_embed(v)  # (B, n, n)
    P = torch.bmm(torch.bmm(U, A), V)

    return P, U, V


class ManifoldHyperConnectionFuse(nn.Module):
    """
    h: hyper hidden matrix (BxLxNxD)
        B: batch_size
        L: Seq_len
        N: expansion rate
        D: feature dim
    """

    def __init__(self, dim, rate, layer_id, max_sk_it=20):
        super(ManifoldHyperConnectionFuse, self).__init__()

        self.n = rate
        self.dim = dim

        self.nc = self.n * self.dim
        self.n2 = self.n * self.n

        # norm flatten
        """
        Observing that RMSNorm in \\mhcshort{} imposes significant latency when operating on 
        the high-dimensional hidden state $\\vec{\\mathbf{x}}_l \\in \\mathbb{R}^{1\\times nC}$, 
        we reorder the dividing-by-norm operation to follow the matrix multiplication. 
        This optimization maintains mathematical equivalence while improving efficiency.
        """
        self.norm = RMSNorm(dim * rate)

        # parameters
        self.w = nn.Parameter(torch.zeros(self.nc, self.n2 + 2 * self.n))
        self.alpha = nn.Parameter(torch.ones(3) * 0.01)
        self.beta = nn.Parameter(torch.zeros(self.n2 + 2 * self.n) * 0.01)

        # max sinkhorn knopp iterations
        self.max_sk_it = max_sk_it

    def mapping(self, h, res_norm):
        B, L, N, D = h.shape

        # 1.vectorize
        h_vec_flat = h.reshape(B, L, N * D)

        # RMSNorm Fused Trick: gamma-scaling
        h_vec = self.norm.gamma * h_vec_flat

        # 2.projection
        H = h_vec @ self.w  # [B, L, n²+2n]

        # RMSNorm Fused: compute r
        r = h_vec_flat.norm(dim=-1, keepdim=True) / math.sqrt(self.nc)
        r_ = 1.0 / r

        # 4. mapping
        n = N
        H_pre = r_ * H[:, :, :n] * self.alpha[0] + self.beta[:n]
        H_post = r_ * H[:, :, n : 2 * n] * self.alpha[1] + self.beta[n : 2 * n]
        H_res = r_ * H[:, :, 2 * n :] * self.alpha[2] + self.beta[2 * n :]

        # 5. final constrained mapping
        H_pre = F.sigmoid(H_pre)  # [B, L, n]
        H_post = 2 * F.sigmoid(H_post)  # [B, L, n]

        # 6. sinkhorn_knopp iteration
        H_res = H_res.reshape(B, L, N, N)
        H_res_exp = H_res.exp()
        with torch.no_grad():
            _, U, V = res_norm(H_res_exp.reshape(B * L, N, N), self.max_sk_it)
        # recover
        P = torch.bmm(torch.bmm(U.detach(), H_res_exp.reshape(B * L, N, N)), V.detach())
        H_res = P.reshape(B, L, N, N)

        return H_pre, H_post, H_res

    def process(self, h, H_pre, H_res):
        """
        h: [B, L, N, D]
        H_pre: [B, L, N]
        H_res: [B, L, N, N]
        """
        B, L, N, D = h.shape

        # H_pre @ h: 输入投影
        h_pre = H_pre.unsqueeze(dim=2) @ h  # [B, L, 1, D]

        # H_res @ h: 残差混合
        h_res = H_res @ h  # [B, L, N, D]

        return h_pre, h_res

    def depth_connection(self, h_res, h_out, H_post):
        """
        h_res: [B, L, N, D]
        h_out: [B, L, 1, D] (层输出)
        H_post: [B, L, N]
        """
        # H_post @ h_out: 输出投影回多流
        post_mapping = H_post.unsqueeze(dim=2) @ h_out  # [B, L, N, D]

        # 最终输出 = 残差混合 + 输出投影
        out = post_mapping + h_res  # [B, L, N, D]

        return out


class MoEAttn(nn.Module):
    def __init__(
        self,
        imu_encoder,
        kp_encoder,
        fusion_dim=512,
        imu_embedding_dim=512,
        kp_embedding_dim=256,
        dropout=0.1,
        mhc_rate=2,  # mHC扩展率
    ):
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
            hidden_dim=fusion_dim, num_heads=8, dropout=dropout
        )

        self.experts = nn.ModuleDict(
            {
                "imu": self._build_expert(fusion_dim, fusion_dim),
                "kp": self._build_expert(fusion_dim, fusion_dim),
                "imukp": self._build_expert(fusion_dim, fusion_dim),
                "kpimu": self._build_expert(fusion_dim, fusion_dim),
            }
        )
        self.router_mix = Mlp(fusion_dim, fusion_dim // 2, 10, drop=dropout)

        # 添加mHC模块
        self.mhc = ManifoldHyperConnectionFuse(
            dim=fusion_dim,
            rate=mhc_rate,  # 扩展率，这里设置为2（对应两个模态）
            layer_id=0,  # 可以是固定的，或者根据实际层数调整
            max_sk_it=20,  # 原论文使用20次迭代
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim * 2),
            nn.Linear(fusion_dim * 2, 10),
        )

        self.mhc_rate = mhc_rate

    def _modify_encoders(self):
        # 冻结/解冻特定层参数
        for name, param in self.imu_encoder.named_parameters():
            param.requires_grad = "rnn" in name

        total_layers = len(self.kp_encoder.st_gcn_networks)
        for i, gcn in enumerate(self.kp_encoder.st_gcn_networks):
            for param in gcn.parameters():
                param.requires_grad = i >= total_layers - 2

    def _build_expert(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Conv1d(in_dim, out_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_dim // 2),
            nn.GELU(),
            nn.Conv1d(out_dim // 2, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_dim),
        )

    def forward(self, imu_input: torch.Tensor, kp_input: torch.Tensor):
        imu_features = self.imu_encoder(imu_input)  # (B, imu_embedding_dim)
        kp_features = self.kp_encoder(kp_input)  # (B, kp_embedding_dim)

        imu_proj = self.imu_proj(imu_features)  # (B, fusion_dim)
        kp_proj = self.kp_proj(kp_features)  # (B, fusion_dim)

        kp_enhance_imu, imu_enhance_kp = self.cross_attn(
            imu_proj, kp_proj
        )  # (B,fusion_dim), (B,fusion_dim)

        # 转置为Conv1d所需格式: (B, D, T)
        imu_attn_t = kp_enhance_imu.unsqueeze(-1)  # (B, fusion_dim, 1)
        kp_attn_t = imu_enhance_kp.unsqueeze(-1)  # (B, fusion_dim, 1)
        imu_raw_t = imu_proj.unsqueeze(-1)  # (B, fusion_dim, 1)
        kp_raw_t = kp_proj.unsqueeze(-1)  # (B, fusion_dim, 1)

        # 专家网络处理
        out_imu = self.experts["imu"](imu_raw_t).squeeze(-1)  # (B, fusion_dim)
        out_kp = self.experts["kp"](kp_raw_t).squeeze(-1)  # (B, fusion_dim)
        out_imukp = self.experts["imukp"](kp_attn_t).squeeze(-1)  # (B, fusion_dim)
        out_kpimu = self.experts["kpimu"](imu_attn_t).squeeze(-1)  # (B, fusion_dim)

        # 路由权重计算
        router_logits = torch.stack(
            [
                self.router_mix(imu_proj).mean(-1),  # (B,)
                self.router_mix(kp_proj).mean(-1),
                self.router_mix(kp_enhance_imu).mean(-1),
                self.router_mix(imu_enhance_kp).mean(-1),
            ],
            dim=1,
        )  # (B, 4)

        router_weights = F.softmax(router_logits, dim=1)  # (B, 4)

        # 专家输出堆叠
        expert_outputs = torch.stack(
            [out_imu, out_kp, out_kpimu, out_imukp], dim=1
        )  # (B, 4, fusion_dim)

        # 加权融合
        moe_output = router_weights.unsqueeze(-1) * expert_outputs  # (B, 4, fusion_dim)

        # 重组为两个模态分支
        output = torch.cat(
            [
                (moe_output[:, 0, :] + moe_output[:, 2, :]).unsqueeze(1),  # IMU分支
                (moe_output[:, 1, :] + moe_output[:, 3, :]).unsqueeze(1),  # KP分支
            ],
            dim=1,
        )  # (B, 2, fusion_dim)

        B = imu_proj.shape[0]

        # 1. 准备mHC输入：多流特征 [B, L=1, N=2, D=fusion_dim]
        # 原始特征作为残差输入
        residual_input = torch.stack([imu_proj, kp_proj], dim=1).unsqueeze(
            1
        )  # [B, 1, 2, fusion_dim]

        # 2. 准备层输出：MoE专家输出 [B, L=1, 1, D=fusion_dim]
        # 将两个模态的专家输出合并（平均或其他方式）
        # 这里我们简单地将两个模态的输出取平均作为层输出
        # layer_output = output.mean(dim=1, keepdim=True).unsqueeze(
        #     2
        # )  # [B, 1, 1, fusion_dim]

        # 3. 计算mHC映射
        H_pre, H_post, H_res = self.mhc.mapping(residual_input, sinkhorn_knopp_batched)

        # 4. 处理输入和残差
        h_pre, h_res = self.mhc.process(residual_input, H_pre, H_res)

        # 5. 深度连接（mHC的核心）
        # 注意：这里layer_output作为F(H_pre @ x)的输出
        mhc_output = self.mhc.depth_connection(h_res, output, H_post)

        # 6. 展平最终输出 [B, 1, 2, fusion_dim] -> [B, 2*fusion_dim]
        output_flat = mhc_output.reshape(B, -1)  # [B, 2*fusion_dim]

        s = self.classifier(output_flat)  # (B, 10)
        return s
