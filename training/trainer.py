"""Original Trainer and wrapper; model and optimization logic is preserved."""
from datetime import datetime
import logging
from pathlib import Path
from typing import Literal

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import config
from utils.evaluation import evaluate_model


class Trainer:
    """训练器类，封装模型训练逻辑"""

    def __init__(
        self,
        model,
        model_name: str,
        train_loader,
        val_loader,
        criterion=nn.CrossEntropyLoss(),
        optimizer: optim.Optimizer | None = None,
        num_epochs: int = 50,
        modal: Literal["unimodal", "multimodal"] = "unimodal",
        test_loader=None,
        device: str = "cpu",
        checkpoint_path: Path = Path("checkpoints"),
        patience: int = 10,
        min_delta: float = 1e-4,
        use_early_stopping: bool = False,
        use_scheduler: bool = True,
        scheduler_mode: Literal["min", "max"] = "min",
        enable_train_info: bool = True,
        enable_train_evaluate_val: float = 0.7,
        logger: logging.Logger = config.logger,
    ):

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        self.model_name = model_name
        self.num_epochs = num_epochs
        self.modal = modal
        self.device = device
        self.checkpoint_path = checkpoint_path

        self.patience = patience
        self.min_delta = min_delta
        self.use_early_stopping = use_early_stopping
        self.use_scheduler = use_scheduler
        self.scheduler_mode = scheduler_mode
        self.enable_train_info = enable_train_info
        self.enable_train_evaluate_val = enable_train_evaluate_val
        self.logger = logger

        # 训练状态
        self.best_val_acc = 0.0
        self.best_val_loss = float("inf")
        self.best_test_acc = 0.0
        self.epochs_no_improve = 0
        self.training_finished = False
        self.enable_evaluate = False
        self.current_epoch = 0

        # 检查点路径
        self.checkpoint_path.mkdir(exist_ok=True, parents=True)

        # 移动到设备
        self.model.to(self.device)

        # 学习率调度器
        self.scheduler = None
        if self.use_scheduler:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, scheduler_mode, factor=0.5, patience=5, min_lr=1e-7
            )

        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.model_weight_prefix = f"{self.model_name} {self.timestamp}"

    def _move_to_device(self, x, y):
        """将数据移动到设备"""
        if isinstance(x, dict):
            x = {k: v.to(self.device).float() for k, v in x.items()}
        else:
            x = x.to(self.device).float()
        y = y.to(self.device).long()
        return x, y

    def _forward_pass(self, x):
        """前向传播，支持单模态和多模态输入"""
        if isinstance(x, dict):
            if "imu" in x and "kp" in x:
                return self.model(x["imu"], x["kp"])
            else:
                # 处理其他字典格式
                return self.model(**x)
        else:
            return self.model(x)

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        correct = total = 0
        running_loss = 0.0

        for x, y in self.train_loader:
            x, y = self._move_to_device(x, y)
            self.optimizer.zero_grad()

            output = self._forward_pass(x)
            if isinstance(output, tuple):
                output = output[1]
            loss = self.criterion(output, y)

            running_loss += loss.item() * y.size(0)
            _, pred = output.max(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

            loss.backward()
            self.optimizer.step()

        epoch_loss = running_loss / total if total > 0 else 0
        epoch_acc = correct / total if total > 0 else 0

        return epoch_loss, epoch_acc

    def validate(self):
        """验证模型"""
        self.model.eval()
        total = correct = 0
        running_loss = 0.0

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = self._move_to_device(x, y)
                output = self._forward_pass(x)
                if isinstance(output, tuple):
                    output = output[1]
                loss = self.criterion(output, y)

                running_loss += loss.item() * y.size(0)
                _, pred = output.max(1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        val_loss = running_loss / total if total > 0 else 0
        val_acc = correct / total if total > 0 else 0

        return val_loss, val_acc

    def evaluate_on_test(self):
        """在测试集上评估"""
        test_acc = evaluate_model(
            self.model,
            "",
            self.test_loader,
            train_mode=True,
            device=self.device,
            modal=self.modal,
        )
        return test_acc

    def check_early_stopping(self, val_loss, val_acc):
        """检查是否应该早停"""
        if not self.use_early_stopping:
            return False

        improved = False
        match self.scheduler_mode:
            case "min":
                if val_loss < self.best_val_loss - self.min_delta:
                    self.best_val_loss = val_loss
                    self.epochs_no_improve = 0
                    improved = True
                else:
                    self.epochs_no_improve += 1
            case "max":
                if val_acc > self.best_val_acc + self.min_delta:
                    self.best_val_acc = val_acc
                    self.epochs_no_improve = 0
                    improved = True
                else:
                    self.epochs_no_improve += 1
            case _:
                raise RuntimeError(f"Invalid scheduler_mode {self.scheduler_mode}")

        if self.epochs_no_improve >= self.patience:
            if self.enable_train_info:
                self.logger.info(
                    f"Early stopping triggered after {self.current_epoch + 1} epochs."
                )
            return True

        return False

    def save_checkpoint(self, prefix="best"):
        """保存模型检查点"""
        torch.save(
            self.model.state_dict(),
            self.checkpoint_path / f"{self.model_weight_prefix}-{prefix}.pt",
        )

    def train(self):
        """执行完整的训练过程"""
        try:
            for epoch in range(self.num_epochs):
                self.current_epoch = epoch

                # 训练阶段
                train_loss, train_acc = self.train_epoch()

                if self.enable_train_info:
                    self.logger.info(
                        f"Epoch {epoch+1}, Train acc {train_acc:.4f}, loss {train_loss:.4f}"
                    )

                # 验证阶段
                val_loss, val_acc = self.validate()

                # 保存最佳模型
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.save_checkpoint("best")
                    if self.enable_train_info:
                        self.logger.info(
                            f"Epoch {epoch + 1}, Val acc {val_acc:.4f}, Val loss {val_loss:.4f}, best model saved."
                        )
                else:
                    if self.enable_train_info:
                        self.logger.info(
                            f"Epoch {epoch + 1}, Val acc {val_acc:.4f}, Val loss {val_loss:.4f}"
                        )
                # 检查是否应该开始测试评估
                if (
                    not self.enable_evaluate
                    and val_acc >= self.enable_train_evaluate_val
                ):
                    self.enable_evaluate = True

                # 测试评估
                if self.enable_evaluate and self.test_loader is not None:
                    test_acc = self.evaluate_on_test()

                    if test_acc > self.best_test_acc:
                        self.best_test_acc = test_acc

                    if self.enable_train_info:
                        self.logger.info(f"Epoch {epoch + 1}, Test acc {test_acc:.4f}")

                # 学习率调度
                if self.use_scheduler:
                    if self.scheduler_mode == "min":
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step(val_acc)

                # 早停检查
                if (
                    self.check_early_stopping(val_loss, val_acc)
                    and self.current_epoch >= self.patience - 1
                ):
                    self.training_finished = True
                    break

        except KeyboardInterrupt:
            self.training_finished = True
            self.logger.info("Training interrupted by user.")

        # 保存最新模型
        self.save_checkpoint("latest")

        # 记录训练完成信息
        status = (
            "with early stopping" if self.training_finished else "completed all epochs"
        )
        self.logger.info(
            f"Training finished {status}. "
            f"Best val acc: {self.best_val_acc:.4f}. "
            f"Best test acc: {self.best_test_acc:.4f}"
        )

        return f"{self.model_weight_prefix}-latest.pt"

    def get_best_metrics(self):
        """获取最佳指标"""
        return {
            "best_val_acc": self.best_val_acc,
            "best_val_loss": self.best_val_loss,
            "best_test_acc": self.best_test_acc,
        }


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer: optim.Optimizer,
    num_epochs: int,
    model_name: str,
    mode: Literal["unimodal", "multimodal"] = "unimodal",
    test_loader=None,
    device: str = "cpu",
    checkpoint_path: Path = Path("checkpoints"),
    patience: int = 10,
    min_delta: float = 1e-4,
    use_early_stopping: bool = True,
    use_scheduler: bool = True,
    scheduler_mode: Literal["min", "max"] = "min",
    enable_train_info: bool = True,
    enable_train_evaluate_val: float = 0.7,
    logger: logging.Logger | None = config.logger,
):
    """
    向后兼容的包装函数，使用 Trainer 类
    """
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        model_name=model_name,
        num_epochs=num_epochs,
        modal=mode,
        test_loader=test_loader,
        device=device,
        checkpoint_path=checkpoint_path,
        patience=patience,
        min_delta=min_delta,
        use_early_stopping=use_early_stopping,
        use_scheduler=use_scheduler,
        scheduler_mode=scheduler_mode,
        enable_train_info=enable_train_info,
        enable_train_evaluate_val=enable_train_evaluate_val,
        logger=logger,
    )

    return trainer.train()
