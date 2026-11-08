"""Original standalone-model construction, loading and optional training."""
from pathlib import Path
from typing import Tuple

from sklearn.metrics import f1_score
import torch
from torch import nn, optim

from models.imu import TemporalConvEncoder
from models.skeleton import STGCN_Encoder
from models.unimodal import UniModalityModel
from training.trainer import train_model
from utils.data import get_dataloaders
from utils.evaluation import evaluate_model


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
    checkpoint_path = Path("checkpoints/pretrained")
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
        imu_model.load_state_dict(torch.load(checkpoint_path / imu_pretrain_name, map_location="cpu"))
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
        kp_model.load_state_dict(torch.load(checkpoint_path / kp_pretrain_name, map_location="cpu"))
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
