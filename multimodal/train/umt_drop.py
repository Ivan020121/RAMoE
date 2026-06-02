from datetime import datetime
from pathlib import Path
from typing import Tuple
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from config import config
from model.multimodal.base import BaselineModel
from model.multimodal.umt_drop import ModdropModel
from model.unimodal.base import UniModalityModel
from model.unimodal.imutconv_newdata import TemporalConvEncoder
from model.unimodal.stgcn import STGCN_Encoder
from .base import get_unimodal_models, train_model
from util.torch_utils import (
    evaluate_model,
    fix_random_seed,
    get_dataloaders,
    load_and_split_data,
    multi_evaluate,
)


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
            l2_reg += torch.sum(param**2)
    else:
        # 仅正则化指定层的权重
        for name, param in model.named_parameters():
            if "weight" in name and any(
                layer_name in name for layer_name in target_layers
            ):
                l2_reg += torch.sum(param**2)

    return lambda_l2 * l2_reg


def train_umt(
    imu_pretrained,
    kp_pretrained,
    train_loader,
    val_loader,
    test_loader,
    checkpoint_path=Path("checkpoint/multimodal"),
    device="cuda",
    lr=1e-3,
    num_epochs=20,
    lambda_task=1.0,
    lambda_distill=0.5,
    lambda_l2=1e-4,
    embedding_dim=(512, 256),
    use_scheduler=False,
):
    imu_embedding_dim, kp_embedding_dim = embedding_dim

    mm_model = ModdropModel(
        imu_pretrained,
        kp_pretrained,
        imu_embedding_dim,
        kp_embedding_dim,
        num_classes=10,
    )

    mm_model.to(device)

    optimizer = optim.AdamW(mm_model.parameters(), lr=lr)
    criterion_task = nn.CrossEntropyLoss()
    scheduler = (
        ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=5, min_lr=1e-7)
        if use_scheduler
        else None
    )

    best_val_acc = 0.0
    model_weight_prefix = f"umt_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    enable_moddrop = False

    try:
        for epoch in range(num_epochs):
            running_loss = 0.0
            running_task_loss = 0.0
            running_l2_loss = 0.0
            correct = 0
            total = 0

            mm_model.train()
            for x, y in train_loader:
                imu_input = x["imu"].to(device).float()
                kp_input = x["kp"].to(device).float()
                y = y.to(device).long()

                optimizer.zero_grad()

                if not enable_moddrop and epoch >= 10:
                    enable_moddrop = True

                y_pred = mm_model(
                    imu_input,
                    kp_input,
                    return_features=False,
                    apply_moddrop=enable_moddrop,
                )

                task_loss = criterion_task(y_pred, y) * lambda_task
                l2_loss = l2_regularization(
                    mm_model, lambda_l2, ["imu_layer", "kp_layer", "output_layer"]
                )
                total_loss = task_loss + l2_loss

                total_loss.backward()
                optimizer.step()

                running_loss += total_loss.item()
                running_task_loss += task_loss.item()
                running_l2_loss += l2_loss.item()

                _, predicted = y_pred.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()

            train_acc = 100.0 * correct / total
            avg_loss = running_loss / len(train_loader)

            # validate
            mm_model.eval()
            total = correct = 0
            for x, y in val_loader:
                imu_input = x["imu"].to(device).float()
                kp_input = x["kp"].to(device).float()
                y = y.to(device).long()

                optimizer.zero_grad()

                y_pred = mm_model(
                    imu_input, kp_input, return_features=False, apply_moddrop=False
                )

                task_loss = criterion_task(y_pred, y) * lambda_task
                l2_loss = l2_regularization(
                    mm_model, lambda_l2, ["imu_layer", "kp_layer", "output_layer"]
                )
                val_loss = task_loss + l2_loss

                _, predicted = y_pred.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
            val_acc = 100.0 * correct / total

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    mm_model.state_dict(),
                    checkpoint_path / f"{model_weight_prefix}-best.pt",
                )

            print(
                f"Epoch [{epoch+1}/{num_epochs}], "
                f"Loss: {avg_loss:.4f}, "
                f"Task Loss: {running_task_loss/len(train_loader):.4f}, "
                f"L2 Loss: {running_l2_loss/len(train_loader):.4f}"
            )
            print(f"Train Acc: {train_acc:.2f}%, " f"Val Acc: {val_acc:.2f}%")

            if use_scheduler:
                scheduler.step(val_loss.detach())

            if val_acc > 0.7:
                test_acc = evaluate_model(
                    mm_model,
                    "",
                    test_loader,
                    train_mode=True,
                    device=device,
                    modal="multimodal",
                )
                print(f"Test Acc: {test_acc*100:.2f}%")
    except KeyboardInterrupt:
        pass

    return model_weight_prefix, mm_model


if __name__ == "__main__":
    logger = config.logger
    seed = 3407
    imu_channels = 9
    train_epoch = 50
    repeat_train = 1
    checkpoint_path = Path("checkpoint/multimodal")
    device = "cuda:0"
    batch_size = 256
    lr = 1e-3
    model_name = "UMT_Moddrop"

    logger.info(f"seed = {seed}")
    fix_random_seed(seed)
    train_set, val_set, test_set = load_and_split_data(
        Path("dataset"),
        imu_channel=imu_channels,
        load_dataset_path=Path("dataset/dataset.pt"),
        augment={
            "imu": {
                "rotation": {"deg": 30},
                # "jitter": {"sigma": 0.05},
                "scale": {"min": 0.5, "max": 2},
                # "amplitude_warp": {"knot": 10, "sigma": 0.2},
                # "permutation": {"segments": 5},
                # "time_warp": {"factor": 1.2},
                "augment_ratio": 0,
                "only_augmented": False,
            },
            "kp": {
                "scale": {"min": 0.5, "max": 1.5},
                "translation": {
                    "max": 0.5,
                },
                "augment_ratio": 1,
                "only_augmented": False,
            },
        },
    )
    mode = "multimodal"
    imu_embedding_dim = 512
    kp_embedding_dim = 256

    for i in range(repeat_train):
        # imu_pretrained, kp_pretrained = train_unimodal_models(train_set, val_set, test_set,
        #     pretrained_names=(f"unimodal_imu 2025-12-02 20:28:31-best.pt",f"unimodal_kp 2025-12-02 20:28:56-best.pt"))
        imu_pretrained, kp_pretrained = get_unimodal_models(
            train_set,
            val_set,
            test_set,
            device=device,
            pretrained_names=(
                "unimodal_imu 2026-01-08 15:58:38-latest.pt",
                "unimodal_kp 2026-01-06 21:00:06-latest.pt",
            ),
        )
        train_loader, val_loader, test_loader = get_dataloaders(
            train_set, val_set, test_set, batch_size=batch_size, data_type="both"
        )
        model_weight_prefix, model = train_umt(
            imu_pretrained,
            kp_pretrained,
            train_loader,
            val_loader,
            test_loader,
            num_epochs=train_epoch,
            use_scheduler=True,
            device=device,
        )

        multi_evaluate(
            model,
            model_name,
            test_loader,
            device=device,
            modal="multimodal",
            checkpoint_best_path=checkpoint_path / f"{model_weight_prefix}-best.pt",
        )
