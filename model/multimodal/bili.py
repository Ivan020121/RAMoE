import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

class GatedMultiheadAttention(nn.Module):
    """
    带门控机制的多头注意力模块（G1位置：SDPA输出后门控）
    支持自注意力(Self-Attention)和交叉注意力(Cross-Attention)
    采用头特定、乘性、sigmoid门控
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        gate_input_dim: Optional[int] = None,
        use_flash_attention: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # 检查头维度是否合理
        assert self.head_dim * num_heads == embed_dim, "embed_dim必须能被num_heads整除"
        
        # 标准多头注意力的QKV投影
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(kdim if kdim else embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(vdim if vdim else embed_dim, embed_dim, bias=bias)
        
        # 输出投影层
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # 门控机制相关参数
        self.gate_input_dim = gate_input_dim or embed_dim
        self.gate_activation = nn.Sigmoid()
        
        # 门控投影层：将门控输入投影到合适维度
        self.gate_proj = nn.Linear(self.gate_input_dim, num_heads * 1)  # 每个头一个标量
        
        self.dropout = dropout
        self.use_flash_attention = use_flash_attention
        
        self.reset_parameters()
    
    def reset_parameters(self):
        # 初始化QKV投影层
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.)
            nn.init.constant_(self.k_proj.bias, 0.)
            nn.init.constant_(self.v_proj.bias, 0.)
        
        # 初始化输出投影层
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)
        
        # 初始化门控投影层
        nn.init.normal_(self.gate_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.gate_proj.bias)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        gate_input: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        return_gate_scores: bool = False,  # 新增：是否返回门控分数用于分析
    ) -> Union[
        torch.Tensor, 
        Tuple[torch.Tensor, Optional[torch.Tensor]],
        Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]
    ]:
        """
        前向传播，支持自注意力和交叉注意力
        
        Args:
            query: [batch_size, tgt_seq_len, embed_dim] 目标序列
            key: [batch_size, src_seq_len, embed_dim] 源序列（自注意力时与query相同）
            value: [batch_size, src_seq_len, embed_dim] 源序列（自注意力时与query相同）
            gate_input: 门控输入，通常为预归一化后的隐藏状态
                       [batch_size, tgt_seq_len, gate_input_dim]
                       如果为None，则使用query作为门控输入
            is_causal: 是否为因果注意力（用于解码器自注意力）
            return_gate_scores: 是否返回门控分数用于分析
            
        Returns:
            output: 注意力输出 [batch_size, tgt_seq_len, embed_dim]
            attn_weights: 注意力权重 [batch_size, num_heads, tgt_seq_len, src_seq_len] (如果need_weights=True)
            gate_scores: 门控分数 [batch_size, num_heads, tgt_seq_len, head_dim] (如果return_gate_scores=True)
        """
        batch_size, tgt_len, embed_dim = query.shape
        src_len = key.shape[1]
        
        # 如果未提供门控输入，使用query作为输入
        if gate_input is None:
            gate_input = query
        
        # 1. QKV投影
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # 2. 重塑为多头形状 [batch_size, seq_len, num_heads, head_dim]
        q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, src_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, src_len, self.num_heads, self.head_dim)
        
        # 3. 转置为 [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)  # [B, H, T, D_h]
        k = k.transpose(1, 2)  # [B, H, S, D_h]
        v = v.transpose(1, 2)  # [B, H, S, D_h]
        
        # 4. 处理注意力掩码
        # 如果提供了key_padding_mask，转换为注意力掩码格式
        if key_padding_mask is not None:
            # key_padding_mask: [batch_size, src_len]
            # 扩展为注意力掩码形状: [batch_size, 1, tgt_len, src_len]
            if attn_mask is None:
                attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            else:
                # 合并掩码
                attn_mask = attn_mask & key_padding_mask.unsqueeze(1).unsqueeze(2)
        
        # 5. 使用PyTorch的scaled_dot_product_attention
        dropout_p = self.dropout if self.training else 0.0
        
        # 设置SDPA后端（可选）
        if self.use_flash_attention and torch.cuda.is_available():
            try:
                from torch.nn.attention import sdpa_kernel, SDPBackend
                
                # 使用Flash Attention后端
                with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                    attn_output = F.scaled_dot_product_attention(
                        q, k, v,
                        attn_mask=attn_mask,
                        dropout_p=dropout_p,
                        is_causal=is_causal,
                        scale=None,  # 使用默认缩放因子
                    )
            except ImportError:
                # 回退到默认实现
                attn_output = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=None,
                )
        else:
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=None,
            )
        
        # 6. 应用门控机制（G1位置）
        # 计算门控分数：σ(gate_input * W_gate)
        gate_scores = self.gate_proj(gate_input)  # [B, T, H]
        
        # 重塑和转置以匹配注意力头的维度
        gate_scores = gate_scores.view(batch_size, tgt_len, self.num_heads, 1)  # [B, T, H, 1]
        gate_scores = self.gate_activation(gate_scores)
        gate_scores = gate_scores.transpose(1, 2)  # [B, H, T, 1]
        
        # 扩展门控分数以匹配注意力输出的每个维度
        gate_scores = gate_scores.expand(-1, -1, -1, self.head_dim)  # [B, H, T, D_h]
        
        # 乘性门控：注意力输出 * 门控分数
        gated_attn_output = attn_output * gate_scores
        
        # 7. 转置回 [batch_size, seq_len, num_heads, head_dim]
        gated_attn_output = gated_attn_output.transpose(1, 2).contiguous()  # [B, T, H, D_h]
        
        # 8. 拼接多头输出并投影
        gated_attn_output = gated_attn_output.view(batch_size, tgt_len, embed_dim)  # [B, T, D]
        output = self.out_proj(gated_attn_output)
        
        # 处理返回值
        returns = []
        returns.append(output)
        
        # 如果需要返回注意力权重
        if need_weights:
            # 手动计算注意力权重
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    attn_weights = attn_weights.masked_fill(~attn_mask, float('-inf'))
                else:
                    attn_weights = attn_weights + attn_mask
            
            attn_weights = F.softmax(attn_weights, dim=-1)
            
            if average_attn_weights:
                attn_weights = attn_weights.mean(dim=1)  # 平均所有头
            
            returns.append(attn_weights)
        else:
            returns.append(None)
        
        # 如果需要返回门控分数
        if return_gate_scores:
            returns.append(gate_scores)
        
        if len(returns) == 1:
            return returns[0]
        else:
            return tuple(returns)


class GatedAttentionLayer(nn.Module):
    """
    完整的门控注意力层，包含层归一化和残差连接
    支持自注意力和交叉注意力
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        pre_norm: bool = True,
        use_flash_attention: bool = True,
        is_cross_attention: bool = False,  # 新增：是否为交叉注意力层
        **kwargs
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.is_cross_attention = is_cross_attention
        
        # 自注意力的归一化
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # 如果是交叉注意力，还需要对key/value进行归一化
        if is_cross_attention:
            self.norm_kv = nn.LayerNorm(embed_dim)
        
        # 门控多头注意力
        self.attn = GatedMultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_flash_attention=use_flash_attention,
            **kwargs
        )
        
        # 前馈网络（自注意力层使用）
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,  # 新增：交叉注意力的上下文输入
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        前向传播，支持自注意力和交叉注意力
        
        Args:
            x: 输入张量 [batch_size, seq_len, embed_dim] (query)
            context: 上下文张量 [batch_size, context_len, embed_dim] (key, value)
                    如果为None，则为自注意力
            is_causal: 是否为因果注意力（仅自注意力有效）
        """
        residual = x
        
        # 预归一化（如果启用）
        if self.pre_norm:
            x_norm = self.norm1(x)
        else:
            x_norm = x
        
        # 决定是自注意力还是交叉注意力
        if context is None:
            # 自注意力：query, key, value都来自x
            key = value = x_norm
            gate_input = x_norm
        else:
            # 交叉注意力：query来自x，key和value来自context
            if self.pre_norm:
                context_norm = self.norm_kv(context)
            else:
                context_norm = context
            key = value = context_norm
            gate_input = x_norm  # 门控输入使用query的归一化版本
        
        # 注意：交叉注意力时不使用因果掩码
        if self.is_cross_attention:
            is_causal = False
        
        # 注意力层
        attn_output, _ = self.attn(
            query=x_norm,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            gate_input=gate_input,
            is_causal=is_causal,
            **kwargs
        )
        
        # 残差连接
        if self.pre_norm:
            x = residual + self.dropout(attn_output)
        else:
            x = self.norm1(residual + self.dropout(attn_output))
        
        # 前馈网络（仅自注意力层需要）
        if not self.is_cross_attention:
            residual = x
            
            if self.pre_norm:
                x_norm = self.norm2(x)
                ffn_output = self.ffn(x_norm)
                x = residual + self.dropout(ffn_output)
            else:
                ffn_output = self.ffn(x)
                x = self.norm2(residual + self.dropout(ffn_output))
        
        return x


class GatedTransformerBlock(nn.Module):
    """
    完整的Transformer块，包含自注意力和交叉注意力
    适用于编码器-解码器架构
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        pre_norm: bool = True,
        use_flash_attention: bool = True,
        **kwargs
    ):
        super().__init__()
        
        # 自注意力层
        self.self_attn = GatedAttentionLayer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            pre_norm=pre_norm,
            use_flash_attention=use_flash_attention,
            is_cross_attention=False,
            **kwargs
        )
        
        # 交叉注意力层
        self.cross_attn = GatedAttentionLayer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            pre_norm=pre_norm,
            use_flash_attention=use_flash_attention,
            is_cross_attention=True,
            **kwargs
        )
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
        self_key_padding_mask: Optional[torch.Tensor] = None,
        cross_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 解码器输入 [batch_size, tgt_seq_len, embed_dim]
            encoder_output: 编码器输出 [batch_size, src_seq_len, embed_dim]
            self_attn_mask: 自注意力掩码
            cross_attn_mask: 交叉注意力掩码
            is_causal: 自注意力是否为因果注意力
        """
        # 自注意力
        x = self.self_attn(
            x=x,
            context=None,
            attn_mask=self_attn_mask,
            key_padding_mask=self_key_padding_mask,
            is_causal=is_causal,
            **kwargs
        )
        
        # 交叉注意力（如果有编码器输出）
        if encoder_output is not None:
            x = self.cross_attn(
                x=x,
                context=encoder_output,
                attn_mask=cross_attn_mask,
                key_padding_mask=cross_key_padding_mask,
                is_causal=False,  # 交叉注意力不使用因果掩码
                **kwargs
            )
        
        return x


# 使用示例
if __name__ == "__main__":
    # 参数设置
    batch_size = 4
    src_len = 64
    tgt_len = 32
    embed_dim = 512
    num_heads = 8
    
    print("=== 测试自注意力 ===")
    # 创建输入
    x = torch.randn(batch_size, tgt_len, embed_dim)
    
    # 创建自注意力层
    self_attn_layer = GatedAttentionLayer(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=0.1,
        pre_norm=True,
        use_flash_attention=False
    )
    
    # 前向传播（自注意力）
    self_output = self_attn_layer(x, is_causal=True)
    print(f"自注意力输入形状: {x.shape}")
    print(f"自注意力输出形状: {self_output.shape}")
    
    print("\n=== 测试交叉注意力 ===")
    # 创建编码器输出（上下文）
    encoder_output = torch.randn(batch_size, src_len, embed_dim)
    
    # 创建交叉注意力层
    cross_attn_layer = GatedAttentionLayer(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=0.1,
        pre_norm=True,
        use_flash_attention=False,
        is_cross_attention=True
    )
    
    # 前向传播（交叉注意力）
    cross_output = cross_attn_layer(x, context=encoder_output)
    print(f"解码器输入形状: {x.shape}")
    print(f"编码器输出形状: {encoder_output.shape}")
    print(f"交叉注意力输出形状: {cross_output.shape}")
    
    print("\n=== 测试完整的Transformer块 ===")
    # 创建完整的Transformer块
    transformer_block = GatedTransformerBlock(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=0.1,
        pre_norm=True,
        use_flash_attention=False
    )
    
    # 前向传播
    block_output = transformer_block(
        x=x,
        encoder_output=encoder_output,
        is_causal=True
    )
    print(f"Transformer块输出形状: {block_output.shape}")
    
    print("\n=== 测试返回门控分数 ===")
    # 测试返回门控分数
    gate_output, _, gate_scores = cross_attn_layer.attn(
        query=x,
        key=encoder_output,
        value=encoder_output,
        need_weights=False,
        return_gate_scores=True
    )
    print(f"门控分数形状: {gate_scores.shape}")
    print(f"门控分数稀疏度: {(gate_scores < 0.1).float().mean().item():.2%}")
    
    print("\n=== 测试掩码功能 ===")
    # 测试带掩码的注意力
    key_padding_mask = torch.zeros(batch_size, src_len, dtype=torch.bool)
    key_padding_mask[:, -10:] = True  # 屏蔽最后10个token
    
    masked_output = cross_attn_layer(
        x=x,
        context=encoder_output,
        key_padding_mask=key_padding_mask
    )
    print(f"带掩码注意力输出形状: {masked_output.shape}")