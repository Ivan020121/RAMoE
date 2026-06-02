from datetime import datetime
import logging
from pathlib import Path
from typing import Literal

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F


from config import config
from model.multimodal.moddrop import ModdropModel
from multimodal.train.base import Trainer
from multimodal.train.umt import train_unimodal_models
from util.torch_utils import evaluate_model, fix_random_seed, get_dataloaders, load_and_split_data

# class ModdropTrainer(Trainer):    
#     def __init__(
#         self,
#         model,
#         model_name: str,
#         train_loader,
#         val_loader,
#         criterion=nn.CrossEntropyLoss(),
#         optimizer: optim.Optimizer|None = None,
#         num_epochs: int = 50,
#         test_loader=None,
#         device: str = "cpu",
#         checkpoint_path: Path = Path("checkpoint"),
#         use_scheduler: bool = True,
#         enable_train_evaluate_val: float = 0.7,
#         # 多模态特有参数
#         gamma_schedule: Literal["linear", "step", "adaptive"] = "linear",
#         gamma_init: float = 0.0,
#         gamma_end: float = 1.0,
#         warmup_epochs: int = 5,
#         freeze_epochs: int = 3,
#     ):
#         super().__init__(
#             model=model,
#             train_loader=train_loader,
#             val_loader=val_loader,
#             criterion=criterion,
#             optimizer=optimizer,
#             model_name=model_name,
#             num_epochs=num_epochs,
#             modal="multimodal",
#             test_loader=test_loader,
#             device=device,
#             checkpoint_path=checkpoint_path,
#             use_scheduler=use_scheduler,
#             enable_train_evaluate_val=enable_train_evaluate_val,
#         )
        
#         self.gamma_schedule = gamma_schedule
#         self.gamma_init = gamma_init
#         self.gamma_end = gamma_end
#         self.warmup_epochs = warmup_epochs
#         self.freeze_epochs = freeze_epochs
        
#         self._init_gamma_control()
        
#     def _init_gamma_control(self):
#         """初始化γ控制参数"""
#         if hasattr(self.model, 'gamma'):
#             # 设置初始gamma值
#             self.model.gamma.data = torch.tensor(self.gamma_init).to(self.device)
#         else:
#             raise RuntimeError("Model does not have gamma parameter. Gamma control disabled.")
    
#     def _update_gamma(self, epoch):
#         """根据调度策略更新gamma参数"""
#         if not hasattr(self.model, 'gamma') or self.gamma_schedule is None:
#             return
        
#         if epoch < self.freeze_epochs:
#             # 冻结期：gamma保持为0
#             new_gamma = 0.0
#         elif self.gamma_schedule == "linear":
#             # 线性增长
#             progress = min((epoch - self.freeze_epochs) / (self.num_epochs - self.freeze_epochs), 1.0)
#             new_gamma = self.gamma_init + (self.gamma_end - self.gamma_init) * progress
#         elif self.gamma_schedule == "step":
#             # 分段增长
#             if epoch < self.warmup_epochs:
#                 new_gamma = 0.0
#             elif epoch < self.warmup_epochs*1.5:
#                 new_gamma = 0.3
#             elif epoch < self.warmup_epochs*2:
#                 new_gamma = 0.6
#             else:
#                 new_gamma = 1.0
#         elif self.gamma_schedule == "adaptive":
#             # 自适应增长（根据验证准确率）
#             if hasattr(self, 'best_val_acc') and self.best_val_acc > 0.6:
#                 # 当准确率较好时，更快地增加gamma
#                 progress = min((epoch - self.freeze_epochs) / (max(10, self.num_epochs // 2)), 1.0)
#                 new_gamma = min(self.gamma_init + progress * 1.5, 1.0)
#             else:
#                 # 否则缓慢增加
#                 progress = min((epoch - self.freeze_epochs) / self.num_epochs, 1.0)
#                 new_gamma = self.gamma_init + (self.gamma_end - self.gamma_init) * progress
#         else:
#             # 默认线性
#             progress = min(epoch / self.num_epochs, 1.0)
#             new_gamma = self.gamma_init + (self.gamma_end - self.gamma_init) * progress
        
#         # 更新gamma值
#         self.model.gamma.data = torch.tensor(new_gamma).to(self.device)
    
#     def _apply_gradient_mask(self):
#         """应用梯度掩码：在训练初期冻结非对角线权重"""
#         if not hasattr(self.model, 'shared_hidden'):
#             return
        
#         if hasattr(self.model, 'gamma') and self.model.gamma.item() < 1e-3:
#             # gamma接近0时，冻结跨模态连接
#             with torch.no_grad():
#                 w = self.model.shared_hidden.weight
#                 imu_dim = self.model.modality_dims[0] if hasattr(self.model, 'modality_dims') else 0
#                 N = self.model.num_classes if hasattr(self.model, 'num_classes') else 10
                
#                 if w.grad is not None:
#                     # 创建梯度掩码
#                     grad_mask = torch.ones_like(w.grad)
                    
#                     # 计算实际维度
#                     if imu_dim > 0 and N > 0:
#                         kp_dim = w.size(1) - imu_dim
                        
#                         # 冻结非对角线分块的梯度
#                         # 模态1到模态2的连接
#                         grad_mask[:N, imu_dim:imu_dim+kp_dim] = 0.0
#                         # 模态2到模态1的连接
#                         grad_mask[N:, :imu_dim] = 0.0
                        
#                         # 应用掩码
#                         w.grad *= grad_mask
    
#     def _compute_cross_modal_grad_norm(self):
#         """计算跨模态连接的梯度范数，用于监控"""
#         if not hasattr(self.model, 'shared_hidden'):
#             return 0.0
        
#         w = self.model.shared_hidden.weight
#         if w.grad is None:
#             return 0.0
        
#         imu_dim = self.model.modality_dims[0] if hasattr(self.model, 'modality_dims') else 0
#         N = self.model.num_classes if hasattr(self.model, 'num_classes') else 10
        
#         if imu_dim == 0 or N == 0:
#             return 0.0
        
#         kp_dim = w.size(1) - imu_dim
        
#         # 提取跨模态梯度
#         cross_grad_1 = w.grad[:N, imu_dim:imu_dim+kp_dim]  # imu→kp
#         cross_grad_2 = w.grad[N:, :imu_dim]  # kp→imu
        
#         # 计算平均梯度范数
#         grad_norm = (cross_grad_1.norm().item() + cross_grad_2.norm().item()) / 2
#         self.cross_modal_grad_norms.append(grad_norm)
        
#         return grad_norm
    
#     def train_epoch(self):
#         self.model.train()
#         correct = total = 0
#         running_loss = 0.0
        
#         # 更新gamma参数
#         self._update_gamma(self.current_epoch)
        
#         for x, y in self.train_loader:
#             x, y = self._move_to_device(x, y)
#             self.optimizer.zero_grad()
            
#             output = self._forward_pass(x)
#             loss = self.criterion(output, y)
#             loss.backward()
#             self._apply_gradient_mask()
            
#             self.optimizer.step()
            
#             running_loss += loss.item() * y.size(0)
#             _, pred = output.max(1)
#             correct += (pred == y).sum().item()
#             total += y.size(0)
        
#         epoch_loss = running_loss / total if total > 0 else 0
#         epoch_acc = correct / total if total > 0 else 0
        
#         if self.enable_train_info:
#             gamma_value = self.model.gamma.item()
#             self.logger.info(f"Epoch {self.current_epoch}: Gamma={gamma_value:.4f}")
        
#         return epoch_loss, epoch_acc
    
def l2_regularization(model, lambda_l2, target_layers=None):
    """
    计算指定层的L2正则化
    
    Args:
        model: 模型
        lambda_l2: 正则化系数
        target_layers: 指定要正则化的层（如['linear', 'fc1']），None表示所有层
    """
    l2_reg = 0.0
    
    if target_layers is None:
        # 正则化所有参数
        for param in model.parameters():
            l2_reg += torch.sum(param ** 2)
    else:
        # 仅正则化指定层的权重
        for name, param in model.named_parameters():
            if 'weight' in name and any(layer_name in name for layer_name in target_layers):
                l2_reg += torch.sum(param ** 2)
    
    return lambda_l2 * l2_reg

class ModdropTrainer(Trainer):    
    def __init__(
        self,
        model,
        model_name: str,
        train_loader,
        val_loader,
        criterion=nn.CrossEntropyLoss(),
        optimizer: optim.Optimizer|None = None,
        num_epochs: int = 50,
        test_loader=None,
        device: str = "cpu",
        checkpoint_path: Path = Path("checkpoint"),
        use_scheduler: bool = True,
        enable_train_evaluate_val: float = 0.7,
        # 多模态特有参数
        gamma_schedule: Literal["linear", "step", "adaptive"] = "linear",
        gamma_init: float = 0.0,
        gamma_end: float = 1.0,
        warmup_epochs: int = 5,
        freeze_epochs: int = 3,
        # Moddrop参数
        moddrop_rate: float = 0.5,
        moddrop_schedule: str = "constant",
        # 一致性损失参数
        consistency_lambda: float = 0.1,
        consistency_type: str = "l2",  # cosine, l2, correlation
    ):
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            model_name=model_name,
            num_epochs=num_epochs,
            modal="multimodal",
            test_loader=test_loader,
            device=device,
            checkpoint_path=checkpoint_path,
            use_scheduler=use_scheduler,
            enable_train_evaluate_val=enable_train_evaluate_val,
        )
        
        self.gamma_schedule = gamma_schedule
        self.gamma_init = gamma_init
        self.gamma_end = gamma_end
        self.warmup_epochs = warmup_epochs
        self.freeze_epochs = freeze_epochs
        
        # Moddrop参数
        self.moddrop_rate = moddrop_rate
        self.moddrop_schedule = moddrop_schedule
        self.model.set_moddrop_rate(moddrop_rate)
        
        # 一致性损失参数
        self.consistency_lambda = consistency_lambda
        self.consistency_type = consistency_type
        self.l2_lambda = 0.01
        
        self._init_gamma_control()
        
    def _init_gamma_control(self):
        """初始化γ控制参数"""
        if hasattr(self.model, 'gamma'):
            # 设置初始gamma值
            self.model.gamma.data = torch.tensor(self.gamma_init).to(self.device)
        else:
            raise RuntimeError("Model does not have gamma parameter. Gamma control disabled.")
        
    def _update_gamma(self, epoch):
        """根据调度策略更新gamma参数"""
        if not hasattr(self.model, 'gamma') or self.gamma_schedule is None:
            return
        
        if epoch < self.freeze_epochs:
            # 冻结期：gamma保持为0
            new_gamma = 0.0
        elif self.gamma_schedule == "linear":
            # 线性增长
            progress = min((epoch - self.freeze_epochs) / (self.num_epochs - self.freeze_epochs), 1.0)
            new_gamma = self.gamma_init + (self.gamma_end - self.gamma_init) * progress
        elif self.gamma_schedule == "step":
            # 分段增长
            if epoch < self.warmup_epochs:
                new_gamma = 0.0
            elif epoch < self.warmup_epochs*1.5:
                new_gamma = 0.3
            elif epoch < self.warmup_epochs*2:
                new_gamma = 0.6
            else:
                new_gamma = 1.0
            # new_gamma = 1.0
        elif self.gamma_schedule == "adaptive":
            # 自适应增长（根据验证准确率）
            if hasattr(self, 'best_val_acc') and self.best_val_acc > 0.6:
                # 当准确率较好时，更快地增加gamma
                progress = min((epoch - self.freeze_epochs) / (max(10, self.num_epochs // 2)), 1.0)
                new_gamma = min(self.gamma_init + progress * 1.5, 1.0)
            else:
                # 否则缓慢增加
                progress = min((epoch - self.freeze_epochs) / self.num_epochs, 1.0)
                new_gamma = self.gamma_init + (self.gamma_end - self.gamma_init) * progress
        else:
            # 默认线性
            progress = min(epoch / self.num_epochs, 1.0)
            new_gamma = self.gamma_init + (self.gamma_end - self.gamma_init) * progress
        
        # 更新gamma值
        self.model.gamma.data = torch.tensor(new_gamma).to(self.device)

    def _update_moddrop_rate(self, epoch):
        """根据调度策略更新Moddrop率"""
        if self.moddrop_schedule == "linear_decay":
            # 线性衰减
            progress = min(epoch / self.num_epochs, 1.0)
            new_rate = self.moddrop_rate * (1.0 - progress * 0.8)  # 逐渐减少到原来的20%
        elif self.moddrop_schedule == "step_decay":
            # 分段衰减
            if epoch < self.num_epochs // 3:
                new_rate = self.moddrop_rate
            elif epoch < 2 * self.num_epochs // 3:
                new_rate = self.moddrop_rate * 0.5
            else:
                new_rate = self.moddrop_rate * 0.2
        else:
            # constant
            new_rate = self.moddrop_rate
        
        self.model.set_moddrop_rate(new_rate)
        
        if hasattr(self, 'logger') and self.enable_train_info:
            self.logger.info(f"Epoch {epoch}: Moddrop rate = {new_rate:.3f}")

    def _apply_gradient_mask(self):
        """应用梯度掩码：在训练初期冻结非对角线权重"""
        if not hasattr(self.model, 'shared_hidden'):
            return
        
        if hasattr(self.model, 'gamma') and self.model.gamma.item() < 1e-3:
            # gamma接近0时，冻结跨模态连接
            with torch.no_grad():
                w = self.model.shared_hidden.weight
                imu_dim = self.model.modality_dims[0] if hasattr(self.model, 'modality_dims') else 0
                N = self.model.num_classes if hasattr(self.model, 'num_classes') else 10
                
                if w.grad is not None:
                    # 创建梯度掩码
                    grad_mask = torch.ones_like(w.grad)
                    
                    # 计算实际维度
                    if imu_dim > 0 and N > 0:
                        kp_dim = w.size(1) - imu_dim
                        
                        # 冻结非对角线分块的梯度
                        # 模态1到模态2的连接
                        grad_mask[:N, imu_dim:imu_dim+kp_dim] = 0.0
                        # 模态2到模态1的连接
                        grad_mask[N:, :imu_dim] = 0.0
                        
                        # 应用掩码
                        w.grad *= grad_mask
    def train_epoch(self,):
        self.model.train()
        correct = total = 0
        running_classification_loss = 0.0
        running_l2_loss=0.0
        running_specificity_loss = 0.0
        running_total_loss = 0.0
        
        # 更新gamma参数和Moddrop率
        self._update_gamma(self.current_epoch)
        self._update_moddrop_rate(self.current_epoch)
        
        for x, y in self.train_loader:
            x, y = self._move_to_device(x, y)
            self.optimizer.zero_grad()
            
            # ========== 前向传播（返回特征用于一致性损失） ==========
            output = self.model(
                x['imu'], x['kp'], 
                return_features=False, 
                apply_moddrop=self.current_epoch>12
            )
            
            # ========== 计算分类损失 ==========
            classification_loss = self.criterion(output, y)
            
            l2_loss = l2_regularization(self.model, self.l2_lambda, ['shared_hidden', 'output_layer'])
            
            
            # ========== 总损失 ==========
            total_loss = (
                classification_loss + 
                self.consistency_lambda * l2_loss 
                # 0.1 * self.consistency_lambda * specificity_loss  # 特异性损失权重较小
            )
            
            # ========== 反向传播 ==========
            total_loss.backward()
            self._apply_gradient_mask()
            
            self.optimizer.step()
            
            # ========== 统计 ==========
            running_classification_loss += classification_loss.item() * y.size(0)
            running_l2_loss += l2_loss.item() * y.size(0)
            running_total_loss += total_loss.item() * y.size(0)
            
            _, pred = output.max(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        
        # 记录损失
        epoch_classification_loss = running_classification_loss / total if total > 0 else 0
        epoch_consistency_loss = running_l2_loss / total if total > 0 else 0
        epoch_total_loss = running_total_loss / total if total > 0 else 0
        
        epoch_acc = correct / total if total > 0 else 0
        
        if self.enable_train_info:
            gamma_value = self.model.gamma.item()
            moddrop_rate = self.model.moddrop_rate
            self.logger.info(
                f"Epoch {self.current_epoch}: "
                f"Gamma={gamma_value:.4f}, "
                f"Moddrop={moddrop_rate:.3f}, "
                f"Cls Loss={epoch_classification_loss:.4f}, "
                f"L2 Loss={epoch_consistency_loss:.4f}, "
                f"Total Loss={epoch_total_loss:.4f}"
            )
        
        return epoch_total_loss, epoch_acc
    
    def validate_epoch(self):
        """验证阶段，禁用Moddrop，只计算分类损失"""
        self.model.eval()
        self.model.enable_moddrop(False)
        
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = self._move_to_device(x, y)
                
                # 验证时只计算分类损失
                output = self.model(x[0], x[1], return_features=False, apply_moddrop=False)
                loss = self.criterion(output, y)
                
                val_loss += loss.item() * y.size(0)
                _, pred = output.max(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        val_loss = val_loss / total if total > 0 else 0
        val_acc = correct / total if total > 0 else 0
        
        # 训练时重新启用Moddrop
        self.model.enable_moddrop(True)
        
        return val_loss, val_acc
    
    def evaluate_on_test(self):
        """在测试集上评估"""
        self.model.enable_moddrop(False)
        both = evaluate_model(
            self.model, 
            "", 
            self.test_loader, 
            train_mode=True, 
            device=self.device, 
            modal=self.modal
        )
        imu = evaluate_model(
            self.model, 
            "", 
            self.test_loader, 
            train_mode=True, 
            device=self.device, 
            modal=self.modal,
            mask_modal='kp'
        )
        kp = evaluate_model(
            self.model, 
            "", 
            self.test_loader, 
            train_mode=True, 
            device=self.device, 
            modal=self.modal,
            mask_modal='imu'
        )
        self.model.enable_moddrop(True)
        return both, imu, kp
    
    def train(self):
        """执行完整的训练过程"""
        try:
            for epoch in range(self.num_epochs):
                self.current_epoch = epoch
                
                # 训练阶段
                train_loss, train_acc = self.train_epoch()
                
                if self.enable_train_info:
                    self.logger.info(f"Epoch {epoch}, Train acc {train_acc:.4f}, loss {train_loss:.4f}")
                
                # 验证阶段
                val_loss, val_acc = self.validate()
                
                # 保存最佳模型
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.save_checkpoint("best")
                    if self.enable_train_info:
                        self.logger.info(f"Epoch {epoch}, Val acc {val_acc:.4f}, best model saved.")
                else:
                    if self.enable_train_info:
                        self.logger.info(f"Epoch {epoch}, Val acc {val_acc:.4f}")
                
                # 检查是否应该开始测试评估
                if not self.enable_evaluate and val_acc >= self.enable_train_evaluate_val:
                    self.enable_evaluate = True
                
                # 测试评估
                if self.enable_evaluate and self.test_loader is not None:
                    both, imu, kp = self.evaluate_on_test()
                    
                    if both > self.best_test_acc:
                        self.best_test_acc = both
                    
                    if self.enable_train_info:
                        self.logger.info(f"Epoch {epoch}, Test acc {both:.4f}, Test IMU acc {imu:.4f}, Test KP acc {kp:.4f}")
                
                # 学习率调度
                if self.use_scheduler:
                    if self.scheduler_mode == "min":
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step(val_acc)
                
                # 早停检查
                if self.check_early_stopping(val_loss, val_acc):
                    self.training_finished = True
                    break
        
        except KeyboardInterrupt:
            self.training_finished = True
            self.logger.info("Training interrupted by user.")
        
        # 保存最新模型
        self.save_checkpoint("latest")
        
        # 记录训练完成信息
        status = "with early stopping" if self.training_finished else "completed all epochs"
        self.logger.info(f"Training finished {status}. "
                        f"Best val acc: {self.best_val_acc:.4f}. "
                        f"Best test acc: {self.best_test_acc:.4f}")
        
if __name__ == "__main__":
    logger = config.logger
    seed = 3407
    imu_channels = 9 
    train_epoch = 40
    repeat_train = 1
    checkpoint_path = Path("checkpoint/multimodal")
    device = "cuda"
    batch_size = 256
    lr = 1e-3
    model_name = "Multimodal_Model"

    logger.info(f"seed = {seed}")
    fix_random_seed(seed)
    train_set, val_set, test_set = load_and_split_data(
        Path("dataset"),
        imu_channel=imu_channels,
        augment={
            "imu": {
                "rotation": {"deg": 30}, 
                # "jitter": {"sigma": 0.05},
                "scale": {"min": 0.5, "max": 2},
                # "amplitude_warp": {"knot": 10, "sigma": 0.2},
                # "permutation": {"segments": 5},
                # "time_warp": {"factor": 1.2},
                "augment_ratio": 0, 
                "only_augmented": False
            },
            "kp": {
                "scale": {
                    "min": 0.5, 
                    "max": 1.5
                },
                "translation": {
                    "max": 0.5, 
                },
                "augment_ratio": 1, 
                "only_augmented": False
            }
        }
    )
    mode = "multimodal"
    imu_embedding_dim = 512
    kp_embedding_dim = 256

    for i in range(repeat_train):
        train_loader, val_loader, test_loader = get_dataloaders(
            train_set, val_set, test_set, batch_size=batch_size, data_type="both"
        )

        imu_encoder, kp_encoder = train_unimodal_models(train_set, val_set, test_set, 
            pretrained_names=(f"unimodal_imu 2025-12-02 20:28:31-best.pt",f"unimodal_kp 2025-12-02 20:28:56-best.pt"))

        model = ModdropModel(imu_encoder, kp_encoder, imu_embedding_dim, kp_embedding_dim)
        trainer = ModdropTrainer(model,
                                "Moddrop",
                                train_loader,
                                val_loader,
                                optimizer=optim.AdamW(model.parameters(), lr=lr),
                                num_epochs = train_epoch,
                                test_loader = test_loader,
                                device = device,
                                # modal = mode,
                                checkpoint_path = checkpoint_path,
                                use_scheduler = True,
                                gamma_schedule = "step",
                                gamma_init = 0,
                                gamma_end = 1,
                                warmup_epochs = 4,
                                freeze_epochs = 0)
        trainer.train()
        
        evaluate_model(
            model, f"{model_name}-latest", test_loader, device=device, save_confusion_matrix=False, modal=mode
        )
        evaluate_model(
            model, f"{model_name}-imu_only-latest", test_loader, device=device, save_confusion_matrix=False, modal=mode, mask_modal='kp'
        )
        evaluate_model(
            model, f"{model_name}-kp_only-latest", test_loader, device=device, save_confusion_matrix=False, modal=mode, mask_modal='imu'
        )
        if trainer.model_weight_prefix:
            model.load_state_dict(torch.load(checkpoint_path / f"{trainer.model_weight_prefix}-best.pt"))
            evaluate_model(model, f"{model_name}-best", test_loader, device=device, save_confusion_matrix=False, modal=mode)
            evaluate_model(
                model, f"{model_name}-imu_only-best", test_loader, device=device, save_confusion_matrix=False, modal=mode, mask_modal='kp'
            )
            evaluate_model(
                model, f"{model_name}-kp_only-best", test_loader, device=device, save_confusion_matrix=False, modal=mode, mask_modal='imu'
            )
