from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModdropModel(nn.Module):
    def __init__(self, imu_encoder, kp_encoder, imu_embedding_dim, kp_embedding_dim, num_classes: int = 10):
        super(ModdropModel, self).__init__()
        self.imu_encoder = imu_encoder
        self.kp_encoder = kp_encoder

        self.moddrop_rate = 0.5
        self.moddrop_enabled = True
        
        
        self.imu_layer = nn.Linear(imu_embedding_dim, num_classes)
        self.kp_layer = nn.Linear(kp_embedding_dim, num_classes)
        self.output_layer = nn.Linear(num_classes * 2, num_classes)
    
    def _apply_moddrop(self, imu_features: torch.Tensor, kp_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        应用Moddrop：随机丢弃模态特征
        
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

        random_choice = None
        if torch.all(modal_mask == 0):
            # 如果两个模态都被丢弃，至少保留一个
            random_choice = torch.randint(0, 2, (1,))

        if modal_mask[0].item() < 0.5 or (random_choice is not None and random_choice==0):
            imu_features = torch.zeros_like(imu_features)
        
        if modal_mask[1].item() < 0.5 or (random_choice is not None and random_choice==1):
            kp_features = torch.zeros_like(kp_features)

        return imu_features, kp_features
    
    def set_moddrop_rate(self, rate: float):
        """设置Moddrop丢弃率"""
        self.moddrop_rate = rate
    
    def enable_moddrop(self, enabled: bool = True):
        """启用或禁用Moddrop"""
        self.moddrop_enabled = enabled

    def forward(self, imu_input, kp_input, return_features: bool = False, apply_moddrop: bool = False):
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
        
        if apply_moddrop:
            imu_features, kp_features = self._apply_moddrop(imu_features, kp_features)
        
        imu_rst = F.relu(self.imu_layer(imu_features), inplace=True)
        kp_rst = F.relu(self.kp_layer(kp_features), inplace=True)
        
        concat_rst = torch.concat([imu_rst, kp_rst], dim=1)
        
        output = self.output_layer(concat_rst)
        
        if return_features:
            return output, imu_features_raw, kp_features_raw
        return output