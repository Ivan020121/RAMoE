"""Original evaluation, modal masking and class-probability statistics."""
import logging
from pathlib import Path
from typing import Literal

from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn.functional as F

from config import config


def evaluate_model(
    model,
    model_name: str,
    test_loader,
    num_classes=10,
    labels=None,
    device="cpu",
    modal: Literal["unimodal", "multimodal"] = "unimodal",
    train_mode: bool = False,
    mask_modal="",
    save_confusion_matrix=False,
    with_report=True,
    with_grouped_report=True,
    confusion_matrix_config: dict = {},
    logger: logging.Logger = config.logger,
):
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    label_tags = [
        "Obstacle",
        "Slip",
        "Hole",
        "Unstable",
        "Climb",
        "Step-off",
        "Squat",
        "Kneel",
        "Fall-down",
        "Walk",
    ]
    if labels:
        label_tags = labels
    group_tags = ["Environment", "Violation", "Overexertion", "IER", "Baseline"]
    group_mapping = [0, 0, 0, 0, 1, 1, 2, 2, 3, 4]

    with torch.no_grad():
        for x, y in test_loader:
            if modal == "multimodal":
                x = {k: v.to(device).float() for k, v in x.items()}
                if mask_modal:
                    x[mask_modal] = torch.zeros_like(x[mask_modal])
            else:
                x = x.to(device).float()
            y = y.to(device).long()
            if isinstance(x, dict):
                output = model(x["imu"], x["kp"])
            else:
                output = model(x)

            if isinstance(output, tuple):
                output = output[1]

            _, pred = output.max(1)
            all_preds.append(pred.cpu())
            all_labels.append(y.cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()

    if train_mode:
        return (y_pred == y_true).mean()

    y_true_group = np.array([group_mapping[label] for label in y_true])
    y_pred_group = np.array([group_mapping[label] for label in y_pred])

    # 输出分类报告
    if with_report:
        logger.info(f"Classification Report - {model_name}")
        print(
            classification_report(
                y_true,
                y_pred,
                labels=range(0, num_classes),
                target_names=label_tags,
                digits=3,
                zero_division=0,
            )
        )
    if with_grouped_report:
        logger.info(f"Classification Report - {model_name}-grouped")
        print(
            classification_report(
                y_true_group,
                y_pred_group,
                labels=range(0, 5),
                target_names=group_tags,
                digits=3,
                zero_division=0,
            )
        )

    # 绘制混淆矩阵
    if save_confusion_matrix:
        row_normalize = confusion_matrix_config.get("row_normalize", False)
        fmt = confusion_matrix_config.get("fmt", "png")
        if row_normalize:
            cm = confusion_matrix(y_true, y_pred, labels=range(0, num_classes)).astype(
                np.float32
            )
            cm = cm / cm.sum(axis=1, keepdims=True)
        else:
            cm = confusion_matrix(y_true, y_pred, labels=range(0, num_classes))
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f" if row_normalize else "d",
            cmap="Blues",
            xticklabels=label_tags,
            yticklabels=label_tags,
        )
        plt.title(f"Confusion Matrix - {model_name} Model")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.savefig(f"img/{model_name}-ConfusionMatrix.{fmt}", dpi=300, format=fmt)
        if with_grouped_report:
            cm_group = confusion_matrix(y_true_group, y_pred_group, labels=list(range(5)))
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm_group,
                annot=True,
                fmt=".2f" if row_normalize else "d",
                cmap="Greens",
                xticklabels=group_tags,
                yticklabels=group_tags,
            )
            plt.title(f"Confusion Matrix (Grouped) - {model_name}")
            plt.xlabel("Predicted Group")
            plt.ylabel("True Group")
            plt.tight_layout()
            plt.savefig(
                f"img/{model_name}-ConfusionMatrix-grouped.{fmt}", dpi=300, format=fmt
            )
        plt.close()

    return y_true, y_pred


def multi_evaluate(
    model, model_name, test_loader, device, modal, checkpoint_best_path: str | Path = "", num_classes=10, save_confusion_matrix=True
):
    evaluate_model(
        model,
        f"{model_name}-latest",
        test_loader,
        device=device,
        num_classes=num_classes,
        save_confusion_matrix=save_confusion_matrix,
        modal=modal,
    )
    evaluate_model(
        model,
        f"{model_name}-imu_only-latest",
        test_loader,
        device=device,
        num_classes=num_classes,
        save_confusion_matrix=save_confusion_matrix,
        modal=modal,
        mask_modal="kp",
    )
    evaluate_model(
        model,
        f"{model_name}-kp_only-latest",
        test_loader,
        device=device,
        num_classes=num_classes,
        save_confusion_matrix=save_confusion_matrix,
        modal=modal,
        mask_modal="imu",
    )
    if checkpoint_best_path:
        model.load_state_dict(torch.load(checkpoint_best_path))
        evaluate_model(
            model,
            f"{model_name}-best",
            test_loader,
            device=device,
            num_classes=num_classes,
            save_confusion_matrix=save_confusion_matrix,
            modal=modal,
        )
        evaluate_model(
            model,
            f"{model_name}-imu_only-best",
            test_loader,
            device=device,
            num_classes=num_classes,
            save_confusion_matrix=save_confusion_matrix,
            modal=modal,
            mask_modal="kp",
        )
        evaluate_model(
            model,
            f"{model_name}-kp_only-best",
            test_loader,
            device=device,
            num_classes=num_classes,
            save_confusion_matrix=save_confusion_matrix,
            modal=modal,
            mask_modal="imu",
        )


def class_output(
    model, test_loader, device, modal="unimodal", with_moe_weights=False, mask_modal=""
):
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    all_router = []

    with torch.no_grad():
        for x, y in test_loader:
            if modal == "multimodal":
                x = {k: v.to(device).float() for k, v in x.items()}
                if mask_modal:
                    x[mask_modal] = torch.zeros_like(x[mask_modal])
            else:
                x = x.to(device).float()
            y = y.to(device).long()
            if modal == "multimodal":
                output = model(x["imu"], x["kp"])
            else:
                output = model(x)

            if with_moe_weights:
                output, router_weights = output

            output = F.softmax(output, dim=1)
            all_preds.append(output.cpu())
            all_labels.append(y.cpu())
            if with_moe_weights:
                all_router.append(router_weights.cpu())

    y_pred = torch.cat(all_preds)
    y_true = torch.cat(all_labels)
    if with_moe_weights:
        router = torch.cat(all_router)

    if with_moe_weights:
        return [y_pred[y_true == k].numpy() for k in range(10)], [
            router[y_true == k].numpy() for k in range(10)
        ]

    return [y_pred[y_true == k].numpy() for k in range(10)]


def paried_test(outputA, outputB):
    statistics = []
    pvals = []
    m = 10
    for i in range(m):
        wilcoxonResult = stats.wilcoxon(
            torch.from_numpy(outputA[i][:, i]),
            torch.from_numpy(outputB[i][:, i]),
        )
        statistics.append(wilcoxonResult.statistic.item())
        pvals.append(wilcoxonResult.pvalue.item())

    qvals = fdr(pvals, m)
    return statistics, pvals, qvals


def fdr(pvals, m):
    pvals_arr = np.array(pvals)
    sorted_indices = np.argsort(pvals_arr)
    sorted_pvals = pvals_arr[sorted_indices]
    rank = np.arange(1, m + 1)  # 排名(1到m)
    bh_values = sorted_pvals * m / rank  # 临时校正值

    for i in range(m - 2, -1, -1):
        if bh_values[i] > bh_values[i + 1]:
            bh_values[i] = bh_values[i + 1]

    bh_values = np.minimum(bh_values, 1.0)
    qvals = np.empty_like(bh_values)
    qvals[sorted_indices] = bh_values
    return qvals
