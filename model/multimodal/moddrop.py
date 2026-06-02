from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# class ModdropModel(nn.Module):
#     def __init__(self, imu_encoder, kp_encoder, imu_embedding_dim, kp_embedding_dim, num_classes: int = 10):
#         super(ModdropModel, self).__init__()
#         self.imu_encoder = imu_encoder
#         self.kp_encoder = kp_encoder
        
#         self.modality_dims = [imu_embedding_dim, kp_embedding_dim]
#         self.num_classes = num_classes
#         self.K = 2  # 模态数量
        
#         self._initialize_fusion_layers(imu_embedding_dim, kp_embedding_dim, num_classes)
        
#         # γ参数：控制模态间权重学习，从0开始逐步增加
#         self.gamma = nn.Parameter(torch.tensor(0.0), requires_grad=False)
    
#     def _initialize_fusion_layers(self, imu_dim, kp_dim, num_classes):
#         """初始化共享隐层和输出层"""
#         F_total = imu_dim + kp_dim  # F = ΣFk
#         N = num_classes
#         K = self.K
        
#         # ========== 共享隐层 W1: F × (N*K) ==========
#         # 创建共享隐层，但我们会自定义初始化
#         self.shared_hidden = nn.Linear(F_total, N * K)
        
#         # 初始化W1权重 - 分块对角初始化
#         w1_weight = torch.zeros(N * K, F_total)
        
#         # 模态1（IMU）对应的分块
#         w1_weight[:N, :imu_dim] = self._xavier_init(imu_dim, N).t()
        
#         # 模态2（KP）对应的分块
#         w1_weight[N:, imu_dim:imu_dim + kp_dim] = self._xavier_init(kp_dim, N).t()
        
#         # 非对角线分块保持为0（初始冻结）
#         self.shared_hidden.weight.data = w1_weight.t()  # PyTorch线性层是权重转置
        
#         # # 初始化偏置
#         nn.init.zeros_(self.shared_hidden.bias)
        
#         # ========== 输出层 W2: (N*K) × N ==========
#         self.output_layer = nn.Linear(N * K, N)
        
#         w2_weight = torch.zeros(N, N * K)
        
#         # 每个模态对应的N×N分块初始化为单位矩阵/ K
#         for k in range(K):
#             start_col = k * N
#             end_col = (k + 1) * N
#             identity_block = torch.eye(N) / K
#             w2_weight[:, start_col:end_col] = identity_block
        
#         self.output_layer.weight.data = w2_weight
        
#         # # 初始化偏置
#         nn.init.zeros_(self.output_layer.bias)
    
#     def _xavier_init(self, fan_in, fan_out):
#         """Xavier均匀分布初始化"""
#         scale = torch.sqrt(torch.tensor(6.0 / (fan_in + fan_out)))
#         return torch.rand(fan_in, fan_out) * 2 * scale - scale
    
#     def forward(self, imu_input, kp_input):
#         # 提取各模态特征
#         imu_features = self.imu_encoder(imu_input)
#         kp_features = self.kp_encoder(kp_input)
        
#         # 拼接模态特征 [B, F_total]
#         combined_features = torch.cat((imu_features, kp_features), dim=1)
        
#         # ========== 共享隐层（带有γ控制的掩码） ==========
#         # 创建W1的掩码：对角线分块权重*1 + 非对角线分块权重*gamma
#         B = combined_features.size(0)
#         imu_dim = imu_features.size(1)
#         kp_dim = kp_features.size(1)
#         N = self.num_classes
        
#         # 获取W1权重
#         w1_weight = self.shared_hidden.weight.data
        
#         # 创建掩码
#         w1_mask = torch.ones_like(w1_weight)
        
#         # 非对角线分块：模态间连接乘以gamma
#         # 模态1到模态2的连接（第一模态的输入到第二模态的输出神经元）
#         w1_mask[:N, imu_dim:imu_dim+kp_dim] = self.gamma
        
#         # 模态2到模态1的连接（第二模态的输入到第一模态的输出神经元）
#         w1_mask[N:, :imu_dim] = self.gamma
        
#         # 应用掩码
#         masked_weight = w1_weight * w1_mask
        
#         # 计算共享隐层激活
#         h_shared = F.relu(
#             F.linear(combined_features, masked_weight.t(), self.shared_hidden.bias)
#         )
        
#         # ========== 输出层 ==========
#         output = self.output_layer(h_shared)
        
#         return output
    
#     def update_gamma(self, new_gamma):
#         """逐步增加gamma，允许学习模态间权重"""
#         # 限制在[0, 1]范围内
#         self.gamma.data = torch.clamp(torch.tensor(new_gamma), 0.0, 1.0)
#         return self.gamma.item()

class ModdropModel(nn.Module):
    def __init__(self, imu_encoder, kp_encoder, imu_embedding_dim, kp_embedding_dim, num_classes: int = 10):
        super(ModdropModel, self).__init__()
        self.imu_encoder = imu_encoder
        self.kp_encoder = kp_encoder
        
        # ========== 冻结预训练编码器 ==========
        # for param in self.imu_encoder.parameters():
        #     param.requires_grad = False
        # for param in self.kp_encoder.parameters():
        #     param.requires_grad = False
        
        self.modality_dims = [imu_embedding_dim, kp_embedding_dim]
        self.num_classes = num_classes
        self.K = 2  # 模态数量
        
        # ========== Moddrop参数 ==========
        self.moddrop_rate = 0.5
        self.moddrop_enabled = True
        
        # ========== 模态一致性损失参数 ==========
        self.consistency_lambda = 0.1  # 一致性损失权重
        
        self._initialize_fusion_layers(imu_embedding_dim, kp_embedding_dim, num_classes)
        
        # γ参数：控制模态间权重学习
        self.gamma = nn.Parameter(torch.tensor(0.0), requires_grad=False)
    
    def _initialize_fusion_layers(self, imu_dim, kp_dim, num_classes):
        """初始化共享隐层和输出层"""
        F_total = imu_dim + kp_dim
        N = num_classes
        K = self.K
        
        # ========== 共享隐层 ==========
        self.shared_hidden = nn.Linear(F_total, N * K)
        
        # 初始化W1权重 - 分块对角初始化
        w1_weight = torch.zeros(N * K, F_total)
        w1_weight[:N, :imu_dim] = self._xavier_init(imu_dim, N).t()
        w1_weight[N:, imu_dim:imu_dim + kp_dim] = self._xavier_init(kp_dim, N).t()
        self.shared_hidden.weight.data = w1_weight.t()
        nn.init.zeros_(self.shared_hidden.bias)
        
        # ========== 输出层 ==========
        self.output_layer = nn.Linear(N * K, N)
        
        w2_weight = torch.zeros(N, N * K)
        for k in range(K):
            start_col = k * N
            end_col = (k + 1) * N
            identity_block = torch.eye(N) / K
            w2_weight[:, start_col:end_col] = identity_block
        self.output_layer.weight.data = w2_weight
        nn.init.zeros_(self.output_layer.bias)

    def _xavier_init(self, fan_in, fan_out):
        """Xavier均匀分布初始化"""
        scale = torch.sqrt(torch.tensor(6.0 / (fan_in + fan_out)))
        return torch.rand(fan_in, fan_out) * 2 * scale - scale
    
    def update_gamma(self, new_gamma):
        """逐步增加gamma，允许学习模态间权重"""
        # 限制在[0, 1]范围内
        self.gamma.data = torch.clamp(torch.tensor(new_gamma), 0.0, 1.0)
        return self.gamma.item()
    
    def _apply_moddrop(self, imu_features: torch.Tensor, kp_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        应用Moddrop：随机丢弃模态或模态内的部分特征
        
        Args:
            imu_features: IMU模态特征 [B, D_imu]
            kp_features: 关键点模态特征 [B, D_kp]
        
        Returns:
            imu_dropped: 应用Moddrop后的IMU特征
            kp_dropped: 应用Moddrop后的关键点特征
        """
        
        if not self.moddrop_enabled or self.moddrop_rate <= 0:
            return imu_features, kp_features
        
        modal_mask = torch.bernoulli(torch.tensor([1.0 - self.moddrop_rate] * 2))
        

        random_choice=None
        # if torch.all(modal_mask < 0.5):
        #     # 随机选择一个模态保留
        #     random_choice = torch.randint(0, 2, (1,))
        # # 应用模态级丢弃
        # if modal_mask[0].item() < 0.5 or (random_choice is not None and random_choice==0):
        #     imu_features = torch.zeros_like(imu_features)
        
        # if modal_mask[1].item() < 0.5 or (random_choice is not None and random_choice==1):
        #     kp_features = torch.zeros_like(kp_features)
        
        if not torch.all(modal_mask < 0.5):
            if modal_mask[0].item() < 0.5:
                imu_features = torch.zeros_like(imu_features)
            
            if modal_mask[1].item() < 0.5:
                kp_features = torch.zeros_like(kp_features)
        
        return imu_features, kp_features
    
    def set_moddrop_rate(self, rate: float):
        """设置Moddrop丢弃率"""
        self.moddrop_rate = rate
    
    def enable_moddrop(self, enabled: bool = True):
        """启用或禁用Moddrop"""
        self.moddrop_enabled = enabled

    def forward(self, imu_input, kp_input, return_features: bool = False, apply_moddrop: bool = None):
        """前向传播
        
        Args:
            imu_input: IMU输入
            kp_input: 关键点输入
            return_features: 是否返回特征（用于计算一致性损失）
            apply_moddrop: 是否应用Moddrop
            
        Returns:
            如果return_features=True: (输出, imu特征, kp特征)
            否则: 输出
        """
        # 提取各模态特征
        imu_features = self.imu_encoder(imu_input)
        kp_features = self.kp_encoder(kp_input)
        
        # 保存原始特征用于一致性损失计算
        imu_features_raw = imu_features.detach().clone()
        kp_features_raw = kp_features.detach().clone()
        
        # ========== 应用Moddrop ==========
        if apply_moddrop is None:
            apply_moddrop = self.moddrop_enabled and self.training
        
        if apply_moddrop:
            imu_features, kp_features = self._apply_moddrop(imu_features, kp_features)
        
        # 拼接模态特征
        combined_features = torch.cat((imu_features, kp_features), dim=1)
        
        # ========== 共享隐层（带有γ控制的掩码） ==========
        imu_dim = imu_features.size(1)
        kp_dim = kp_features.size(1)
        N = self.num_classes
        
        w1_weight = self.shared_hidden.weight
        w1_mask = torch.ones_like(w1_weight)
        
        # # 应用gamma掩码
        w1_mask[:N, imu_dim:imu_dim+kp_dim] = self.gamma
        w1_mask[N:, :imu_dim] = self.gamma
        masked_weight = w1_weight * w1_mask
        
        h_shared = F.relu(
            F.linear(combined_features, masked_weight.t(), self.shared_hidden.bias)
        )
        # h_shared = F.relu(self.shared_hidden(combined_features))
        
        # ========== 输出层 ==========
        output = self.output_layer(h_shared)
        
        if return_features:
            return output, imu_features_raw, kp_features_raw
        return output