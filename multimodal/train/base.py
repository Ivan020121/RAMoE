from datetime import datetime
import logging
from pathlib import Path
from typing import Literal, Tuple

from sklearn.metrics import f1_score
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


from config import config
from model.multimodal.base import BaselineModel
from model.unimodal.base import UniModalityModel
from model.unimodal.imutconv import TemporalConvEncoder
from model.unimodal.stgcn import STGCN_Encoder
from util.torch_utils import (
    add_single_modal_samples,
    evaluate_model,
    fix_random_seed,
    get_dataloaders,
    load_and_split_data,
    multi_evaluate,
)


def train_model_bk(
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
    checkpoint_path: Path = Path("checkpoint"),
    patience: int = 10,
    min_delta: float = 1e-4,
    use_early_stopping: bool = True,
    use_scheduler: bool = True,
    scheduler_mode: Literal["min", "max"] = "min",
    enable_train_info: bool = True,
    enable_train_evaluate_val: float = 2,
    logger: logging.Logger = config.logger,
):
    checkpoint_path.mkdir(exist_ok=True)

    model.to(device)
    best_val_acc = 0.0
    best_val_loss = float("inf")
    test_acc = 0
    best_test_acc = 0
    epochs_no_improve = 0
    training_finished = False
    enable_evaluate = False

    scheduler = (
        ReduceLROnPlateau(
            optimizer, scheduler_mode, factor=0.5, patience=5, min_lr=1e-7
        )
        if use_scheduler
        else None
    )
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_weight_prefix = f"{model_name} {timestamp}"
    try:
        for epoch in range(num_epochs):
            model.train()
            correct = total = 0
            running_loss = 0.0

            for x, y in train_loader:
                # Support both tensor batches and dict batches (for multimodal inputs).
                if isinstance(x, dict):
                    x = {k: v.to(device).float() for k, v in x.items()}
                else:
                    x = x.to(device).float()
                y = y.to(device).long()
                optimizer.zero_grad()

                if isinstance(x, dict):
                    output = model(x["imu"], x["kp"])
                else:
                    output = model(x)
                loss = criterion(output, y)

                # use label batch size for loss accumulation (works for tensor or dict batches)
                running_loss += loss.item() * y.size(0)
                _, pred = output.max(1)
                correct += (pred == y).sum().item()
                total += y.size(0)

                loss.backward()
                optimizer.step()

            epoch_loss = running_loss / total
            epoch_acc = correct / total

            if enable_train_info:
                logger.info(
                    f"Epoch {epoch}, Train acc {epoch_acc:.4f}, loss {epoch_loss:.4f}"
                )

            # validete
            model.eval()
            total = correct = 0
            with torch.no_grad():
                for x, y in val_loader:
                    if isinstance(x, dict):
                        x = {k: v.to(device).float() for k, v in x.items()}
                    else:
                        x = x.to(device).float()
                    y = y.to(device).long()

                    if isinstance(x, dict):
                        output = model(x["imu"], x["kp"])
                    else:
                        output = model(x)
                    val_loss = criterion(output, y)

                    _, pred = output.max(1)
                    correct += (pred == y).sum().item()
                    total += y.size(0)

            val_acc = correct / total

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    model.state_dict(),
                    checkpoint_path / f"{model_weight_prefix}-best.pt",
                )
                if enable_train_info:
                    logger.info(
                        f"Epoch {epoch},     Val acc {val_acc:.4f}, best model saved."
                    )
            else:
                if enable_train_info:
                    logger.info(f"Epoch {epoch},     Val acc {val_acc:.4f}")

            if not enable_evaluate and val_acc >= enable_train_evaluate_val:
                enable_evaluate = True

            if enable_evaluate:
                test_acc = evaluate_model(
                    model, "", test_loader, train_mode=True, device=device, modal=mode
                )
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                if enable_train_info:
                    logger.info(f"Epoch {epoch},     Test acc {test_acc:.4f}")

            if use_scheduler:
                scheduler.step(val_loss)

            if use_early_stopping:
                match scheduler_mode:
                    case "min":
                        if val_loss < best_val_loss - min_delta:
                            best_val_loss = val_loss
                            epochs_no_improve = 0
                        else:
                            epochs_no_improve += 1
                    case "max":
                        if val_acc > best_val_acc + min_delta:
                            best_val_acc = val_acc
                            epochs_no_improve = 0
                        else:
                            epochs_no_improve += 1
                    case _:
                        raise RuntimeError(f"Invalid scheduler_mode {scheduler_mode}")

                if epochs_no_improve >= patience:
                    if enable_train_info:
                        logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                    training_finished = True
                    break
    except KeyboardInterrupt:
        training_finished = True

    torch.save(
        model.state_dict(),
        checkpoint_path / f"{model_weight_prefix}-latest.pt",
    )

    status = "with early stopping" if training_finished else "completed all epochs"
    logger.info(
        f"Training finished {status}. Best val acc: {best_val_acc:.4f}. Best test acc {best_test_acc:.4f}"
    )

    return model_weight_prefix


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
        checkpoint_path: Path = Path("checkpoint"),
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
    checkpoint_path: Path = Path("checkpoint"),
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

def test_f1(model, test_loader, device, mode, num_classes=10):
    # 获取各个类别在测试集上的F1-score list
    y_true, y_pred = evaluate_model(
        model,
        "unimodal-f1",
        test_loader,
        device=device,
        modal=mode,
        with_report=False,
        with_grouped_report=False,
    )
    # 计算每个类别的 F1-score
    return f1_score(y_true, y_pred, average=None, labels=list(range(num_classes)))

def get_unimodal_models(
    train_set,
    val_set,
    test_set,
    imu_channels=9,
    imu_embedding_dim=512,
    imu_epoch=20,
    kp_epoch=30,
    kp_embedding_dim=512,
    num_classes=10,
    batch_size=256,
    lr=1e-3,
    no_train: bool = False,
    with_head: bool = False,
    return_both: bool = False,
    pretrained_names: Tuple[str, str] = ("", ""),
    device="cuda",
    with_evaluate_rst=False
):
    mode = "unimodal"
    checkpoint_path = Path("checkpoint/use")
    imu_pretrain_name, kp_pretrain_name = pretrained_names[0], pretrained_names[1]

    imu_encoder = TemporalConvEncoder(
        input_dim=128, size_embeddings=imu_embedding_dim, imu_channels=imu_channels
    )
    kp_encoder = STGCN_Encoder(2, edge_importance_weighting=True)

    imu_model = UniModalityModel(
        imu_encoder,
        imu_embedding_dim,
        num_classes=num_classes,
        with_feature=with_head and return_both,
    )
    kp_model = UniModalityModel(
        kp_encoder,
        kp_embedding_dim,
        num_classes=num_classes,
        with_feature=with_head and return_both,
    )

    if no_train:
        if with_head:
            return imu_model, kp_model

        return imu_encoder, kp_encoder

    train_loader, val_loader, test_loader = get_dataloaders(
        train_set, val_set, test_set, batch_size=batch_size, data_type="imu"
    )

    if imu_pretrain_name:
        imu_model.load_state_dict(torch.load(checkpoint_path / imu_pretrain_name))
        if with_evaluate_rst:
            imu_model.to(device)
            imu_f1 = test_f1(imu_model, test_loader, device, mode, num_classes=num_classes)
            imu_encoder.to("cpu")
    else:
        train_model(
            imu_model,
            train_loader,
            val_loader,
            nn.CrossEntropyLoss(),
            optim.AdamW(imu_model.parameters(), lr=lr),
            test_loader=test_loader,
            num_epochs=imu_epoch,
            model_name="unimodal_imu",
            mode=mode,
            use_early_stopping=False,
            use_scheduler=True,
            patience=10,
            min_delta=5e-5,
            checkpoint_path=checkpoint_path,
            device=device,
            enable_train_evaluate_val=0.7,
            enable_train_info=True,
        )
        if with_evaluate_rst:
            imu_f1 = test_f1(imu_model, test_loader, device, mode, num_classes=num_classes)
        imu_encoder.to("cpu")

    train_loader, val_loader, test_loader = get_dataloaders(
        train_set, val_set, test_set, batch_size=batch_size, data_type="kp"
    )

    if kp_pretrain_name:
        kp_model.load_state_dict(torch.load(checkpoint_path / kp_pretrain_name))
        if with_evaluate_rst:
            kp_model.to(device)
            kp_f1 = test_f1(kp_model, test_loader, device, mode, num_classes=num_classes)
            kp_encoder.to("cpu")
    else:
        train_model(
            kp_model,
            train_loader,
            val_loader,
            nn.CrossEntropyLoss(),
            optim.AdamW(kp_model.parameters(), lr=lr),
            test_loader=test_loader,
            num_epochs=kp_epoch,
            model_name="unimodal_kp",
            mode=mode,
            use_early_stopping=False,
            use_scheduler=True,
            patience=10,
            min_delta=5e-5,
            checkpoint_path=checkpoint_path,
            device=device,
            enable_train_evaluate_val=0.7,
            enable_train_info=True,
        )
        if with_evaluate_rst:
            kp_f1 = test_f1(kp_model, test_loader, device, mode, num_classes=num_classes)
        kp_encoder.to("cpu")

    if with_head:
        if with_evaluate_rst:
            return imu_model, kp_model, {"imu": imu_f1, "kp": kp_f1}
        return imu_model, kp_model
    
    if with_evaluate_rst:
        return imu_encoder, kp_encoder, {"imu": imu_f1, "kp": kp_f1}
    return imu_encoder, kp_encoder


if __name__ == "__main__":
    logger = config.logger
    seed = 3407
    imu_channels = 9
    train_epoch = 40
    repeat_train = 1
    checkpoint_path = Path("checkpoint/multimodal")
    device = "cuda:1"
    batch_size = 256
    lr = 1e-3
    model_name = "baseline"

    logger.info("seed = %s", seed)
    fix_random_seed(seed, True)
    train_set, val_set, test_set = load_and_split_data(
        Path("dataset"),
        imu_channel=imu_channels,
        load_dataset_path=Path("dataset/dataset_ag.pt"),
    )

    modal = "multimodal"
    imu_embedding_dim = 512
    kp_embedding_dim = 512

    for i in range(repeat_train):
        imu_encoder, kp_encoder = get_unimodal_models(
            train_set, val_set, test_set, device=device, no_train=True
        )

        train_loader, val_loader, test_loader = get_dataloaders(
            add_single_modal_samples(train_set),
            val_set,
            test_set,
            batch_size=batch_size,
            data_type="both",
        )
        model = BaselineModel(
            imu_encoder, kp_encoder, imu_embedding_dim, kp_embedding_dim
        )
        trainer = Trainer(
            model,
            "Baseline",
            train_loader,
            val_loader,
            optimizer=optim.AdamW(model.parameters(), lr=lr),
            num_epochs=train_epoch,
            test_loader=test_loader,
            device=device,
            modal=modal,
            checkpoint_path=checkpoint_path,
            use_scheduler=True,
            use_early_stopping=True,
            patience=10,
        )

        trainer.train()

        multi_evaluate(
            model,
            f"{model_name}",
            test_loader,
            device=device,
            modal=modal,
            checkpoint_best_path=checkpoint_path
            / f"{trainer.model_weight_prefix}-best.pt",
        )
