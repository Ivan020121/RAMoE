from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import config
from model.multimodal.base import BaselineModel
from .base import get_unimodal_models
from util.torch_utils import (
    evaluate_model,
    fix_random_seed,
    get_dataloaders,
    load_and_split_data,
    multi_evaluate,
)


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
    embedding_dim=(512, 256),
    use_scheduler=False,
):
    imu_embedding_dim, kp_embedding_dim = embedding_dim
    # froze parameter
    for param in imu_pretrained.parameters():
        param.requires_grad = False
    for param in kp_pretrained.parameters():
        param.requires_grad = False

    mm_model = BaselineModel(
        imu_pretrained,
        kp_pretrained,
        imu_embedding_dim,
        kp_embedding_dim,
        num_classes=10,
    )

    mm_model.to(device)

    optimizer = optim.AdamW(mm_model.parameters(), lr=lr)
    criterion_task = nn.CrossEntropyLoss()
    criterion_distill = nn.MSELoss()
    scheduler = (
        ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=5, min_lr=1e-7)
        if use_scheduler
        else None
    )

    best_val_acc = 0.0
    model_weight_prefix = f"umt_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    try:
        for epoch in range(num_epochs):
            running_loss = 0.0
            running_task_loss = 0.0
            running_distill_loss = 0.0
            correct = 0
            total = 0

            mm_model.train()
            for x, y in train_loader:
                imu_input = x["imu"].to(device).float()
                kp_input = x["kp"].to(device).float()
                y = y.to(device).long()

                optimizer.zero_grad()

                with torch.no_grad():
                    f_pre_imu = imu_pretrained(imu_input)
                    f_pre_kp = kp_pretrained(kp_input)

                y_pred, f_imu, f_kp = mm_model(
                    imu_input, kp_input, return_features=True
                )

                task_loss = criterion_task(y_pred, y) * lambda_task
                distill_loss_imu = criterion_distill(f_imu, f_pre_imu) * lambda_distill
                distill_loss_kp = criterion_distill(f_kp, f_pre_kp) * lambda_distill
                total_loss = task_loss + distill_loss_imu + distill_loss_kp

                total_loss.backward()
                optimizer.step()

                running_loss += total_loss.item()
                running_task_loss += task_loss.item()
                running_distill_loss += distill_loss_imu.item() + distill_loss_kp.item()

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

                with torch.no_grad():
                    f_pre_imu = imu_pretrained(imu_input)
                    f_pre_kp = kp_pretrained(kp_input)

                y_pred, f_imu, f_kp = mm_model(
                    imu_input, kp_input, return_features=True
                )

                task_loss = criterion_task(y_pred, y) * lambda_task
                distill_loss_imu = criterion_distill(f_imu, f_pre_imu) * lambda_distill
                distill_loss_kp = criterion_distill(f_kp, f_pre_kp) * lambda_distill
                val_loss = task_loss + distill_loss_imu + distill_loss_kp

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
                f"Distill Loss: {running_distill_loss/len(train_loader):.4f}"
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
    train_epoch = 10
    repeat_train = 1
    checkpoint_path = Path("checkpoint/multimodal")
    device = "cuda:0"
    batch_size = 256
    lr = 6e-5
    model_name = "UMT"

    logger.info(f"seed = {seed}")
    fix_random_seed(seed)
    train_set, val_set, test_set = load_and_split_data(
        Path("dataset"),
        imu_channel=imu_channels,
        # load_dataset_path=Path("dataset/dataset.pt"),
        # augment={
        #     "imu": {
        #         "rotation": {"deg": 30},
        #         # "jitter": {"sigma": 0.05},
        #         "scale": {"min": 0.5, "max": 2},
        #         # "amplitude_warp": {"knot": 10, "sigma": 0.2},
        #         # "permutation": {"segments": 5},
        #         # "time_warp": {"factor": 1.2},
        #         "augment_ratio": 0,
        #         "only_augmented": False,
        #     },
        #     "kp": {
        #         "scale": {"min": 0.5, "max": 1.5},
        #         "translation": {
        #             "max": 0.5,
        #         },
        #         "augment_ratio": 1,
        #         "only_augmented": False,
        #     },
        # },
    )
    mode = "multimodal"
    imu_embedding_dim = 512
    kp_embedding_dim = 512

    for i in range(repeat_train):
        imu_pretrained, kp_pretrained = get_unimodal_models(
            train_set,
            val_set,
            test_set,
            device=device,
            pretrained_names=(
                "unimodal_imu 2026-01-08 15:58:38-latest.pt",
                "unimodal_kp 2026-01-25 13:00:01-latest.pt",
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
            device=device,
            num_epochs=train_epoch,
        )

        multi_evaluate(
            model,
            model_name,
            test_loader,
            device=device,
            modal="multimodal",
            checkpoint_best_path=checkpoint_path / f"{model_weight_prefix}-best.pt",
        )
