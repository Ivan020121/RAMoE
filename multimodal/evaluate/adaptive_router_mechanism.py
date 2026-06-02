from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from scipy import stats
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import config
from model.multimodal.moe import PRISM
from model.unimodal.base import UniModalityModel
from model.unimodal.imutconv import TemporalConvEncoder
from model.unimodal.stgcn import STGCN_Encoder


LABEL_NAMES = [
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

BLUE_GREEN_CMAP = LinearSegmentedColormap.from_list(
    "blue_green_reference",
    ["#F7FEF0", "#CEEFCC", "#BFE8C1", "#6FCBCA", "#58B8D1", "#3492B2", "#04579B"],
)

BAR_COLORS = ["#04579B", "#3492B2", "#58B8D1", "#6FCBCA", "#92C2A6", "#9ED17B", "#BFE8C1", "#CEEFCC"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adaptive Router mechanism analysis: expert profile and factor effect size."
    )
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "dataset" / "dataset_ag.pt")
    parser.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "checkpoint")
    parser.add_argument("--prism-weight", type=str, default="PRISM 2026-02-01 22_21_27-latest.pt")
    parser.add_argument("--imu-weight", type=str, default="unimodal_imu 2026-01-31 17_49_32-latest.pt")
    parser.add_argument("--kp-weight", type=str, default="unimodal_kp 2026-01-25 13_00_01-latest.pt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--imu-channels", type=int, default=9)
    parser.add_argument("--fusion-dim", type=int, default=512)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "img" / "adaptive_router_mechanism")
    parser.add_argument("--figure-format", choices=["png", "svg", "pdf"], default="png")
    parser.add_argument("--reuse-records", action="store_true")
    return parser.parse_args()


def safe_torch_load(path: Path, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_saved_dataset(dataset_path: Path):
    try:
        return torch.load(dataset_path, weights_only=False)
    except TypeError:
        return torch.load(dataset_path)


def fix_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class MultiModalTensorDataset(torch.utils.data.Dataset):
    def __init__(self, imu_tensor: torch.Tensor, kp_tensor: torch.Tensor, labels: torch.Tensor):
        if len(imu_tensor) != len(kp_tensor) or len(kp_tensor) != len(labels):
            raise ValueError("IMU, KP, and label tensors must have the same length.")
        self.imu = imu_tensor
        self.kp = kp_tensor
        self.labels = labels

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, idx):
        return {"imu": self.imu[idx], "kp": self.kp[idx]}, self.labels[idx]


def build_loader(dataset_tuple, batch_size: int) -> DataLoader:
    imu, kp, labels = dataset_tuple
    return DataLoader(MultiModalTensorDataset(imu, kp, labels), batch_size=batch_size, shuffle=False)


def build_unimodal_models(args: argparse.Namespace):
    imu_encoder = TemporalConvEncoder(
        input_dim=128,
        size_embeddings=args.fusion_dim,
        imu_channels=args.imu_channels,
    )
    kp_encoder = STGCN_Encoder(2, edge_importance_weighting=True)
    imu_model = UniModalityModel(
        imu_encoder,
        args.fusion_dim,
        num_classes=args.num_classes,
        with_feature=True,
    )
    kp_model = UniModalityModel(
        kp_encoder,
        args.fusion_dim,
        num_classes=args.num_classes,
        with_feature=True,
    )
    return imu_model, kp_model


def load_prism(args: argparse.Namespace) -> PRISM:
    imu_model, kp_model = build_unimodal_models(args)
    imu_model.load_state_dict(safe_torch_load(args.checkpoint_dir / args.imu_weight, args.device))
    kp_model.load_state_dict(safe_torch_load(args.checkpoint_dir / args.kp_weight, args.device))
    model = PRISM(
        imu_model,
        kp_model,
        fusion_dim=args.fusion_dim,
        dropout=0.3,
        num_classes=args.num_classes,
        device=args.device,
    )
    model.load_state_dict(safe_torch_load(args.checkpoint_dir / args.prism_weight, args.device))
    model.to(args.device)
    model.eval()
    return model


@torch.no_grad()
def forward_details(model: PRISM, imu: torch.Tensor, kp: torch.Tensor) -> dict[str, torch.Tensor]:
    imu_features, imu_logits = model.imu_encoder(imu)
    kp_features, kp_logits = model.kp_encoder(kp)
    feat = torch.cat([imu_features, kp_features], dim=1)
    router_logits = model.router(feat)
    router_prob = F.softmax(router_logits, dim=1)

    modal_mask = model.modality_selector(imu_logits, kp_logits)
    imu_expert_out = model.imu_expert(imu_features.unsqueeze(-1)).squeeze(-1)
    kp_expert_out = model.kp_expert(kp_features.unsqueeze(-1)).squeeze(-1)
    modal_out = torch.stack([imu_expert_out, kp_expert_out], dim=1) * modal_mask.unsqueeze(-1)
    modal_out = modal_out.flatten(1)

    top_k_logits, indices = router_logits.topk(model.top_k, dim=-1)
    zeros = torch.full_like(router_logits, float("-inf"))
    sparse_logits = zeros.scatter(-1, indices, top_k_logits)
    router_sparse_weights = F.softmax(sparse_logits, dim=1)
    expert_outputs = torch.stack([expert(feat.unsqueeze(-1)) for expert in model.experts], dim=1).squeeze(-1)
    moe_output = (router_sparse_weights.unsqueeze(-1) * expert_outputs).mean(1)
    full_logits = model.classifier(torch.cat([modal_out, moe_output], dim=1))

    return {
        "imu_features": imu_features,
        "kp_features": kp_features,
        "imu_logits": imu_logits,
        "kp_logits": kp_logits,
        "router_prob": router_prob,
        "full_logits": full_logits,
    }


def probability_js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    m = 0.5 * (p + q)
    kl_pm = (p * (torch.log(p.clamp_min(1e-12)) - torch.log(m.clamp_min(1e-12)))).sum(dim=1)
    kl_qm = (q * (torch.log(q.clamp_min(1e-12)) - torch.log(m.clamp_min(1e-12)))).sum(dim=1)
    return 0.5 * (kl_pm + kl_qm)


def prediction_margin(prob: torch.Tensor) -> torch.Tensor:
    top2 = prob.topk(2, dim=1).values
    return top2[:, 0] - top2[:, 1]


@torch.no_grad()
def collect_router_records(model: PRISM, loader, device: str) -> pd.DataFrame:
    rows = []
    sample_offset = 0
    for x, y in tqdm(loader, desc="Collecting test router records"):
        imu = x["imu"].to(device).float()
        kp = x["kp"].to(device).float()
        labels = y.to(device).long()
        details = forward_details(model, imu, kp)

        imu_prob = F.softmax(details["imu_logits"], dim=1)
        kp_prob = F.softmax(details["kp_logits"], dim=1)
        full_prob = F.softmax(details["full_logits"], dim=1)
        router_prob = details["router_prob"]

        imu_pred = imu_prob.argmax(dim=1)
        kp_pred = kp_prob.argmax(dim=1)
        full_pred = full_prob.argmax(dim=1)
        selected_expert = router_prob.argmax(dim=1)

        prob_js = probability_js_divergence(imu_prob, kp_prob)
        prob_l1 = torch.abs(imu_prob - kp_prob).sum(dim=1)
        prob_cosine_distance = 1 - F.cosine_similarity(imu_prob, kp_prob, dim=1)

        imu_features_normed = F.normalize(details["imu_features"], dim=1)
        kp_features_normed = F.normalize(details["kp_features"], dim=1)
        feature_cosine_distance = 1 - F.cosine_similarity(imu_features_normed, kp_features_normed, dim=1)
        feature_l2_distance = torch.linalg.vector_norm(imu_features_normed - kp_features_normed, ord=2, dim=1)
        imu_feature_norm = torch.linalg.vector_norm(details["imu_features"], ord=2, dim=1)
        kp_feature_norm = torch.linalg.vector_norm(details["kp_features"], ord=2, dim=1)

        for i in range(labels.size(0)):
            row = {
                "sample_id": sample_offset + i,
                "label": int(labels[i].item()),
                "label_name": LABEL_NAMES[int(labels[i].item())],
                "imu_pred": int(imu_pred[i].item()),
                "kp_pred": int(kp_pred[i].item()),
                "full_pred": int(full_pred[i].item()),
                "agreement": int(imu_pred[i].item() == kp_pred[i].item()),
                "full_correct": int(full_pred[i].item() == labels[i].item()),
                "selected_expert": int(selected_expert[i].item()),
                "full_confidence": float(full_prob[i].max().item()),
                "full_margin": float(prediction_margin(full_prob)[i].item()),
                "imu_confidence": float(imu_prob[i].max().item()),
                "kp_confidence": float(kp_prob[i].max().item()),
                "confidence_gap": float(torch.abs(imu_prob[i].max() - kp_prob[i].max()).item()),
                "prob_js_divergence": float(prob_js[i].item()),
                "prob_l1_distance": float(prob_l1[i].item()),
                "prob_cosine_distance": float(prob_cosine_distance[i].item()),
                "feature_cosine_distance": float(feature_cosine_distance[i].item()),
                "feature_l2_distance": float(feature_l2_distance[i].item()),
                "imu_feature_norm": float(imu_feature_norm[i].item()),
                "kp_feature_norm": float(kp_feature_norm[i].item()),
                "feature_norm_gap": float(torch.abs(imu_feature_norm[i] - kp_feature_norm[i]).item()),
            }
            for expert_idx in range(router_prob.size(1)):
                row[f"expert_{expert_idx + 1}_prob"] = float(router_prob[i, expert_idx].item())
            rows.append(row)
        sample_offset += labels.size(0)
    return pd.DataFrame(rows)


def expert_profile_analysis(records: pd.DataFrame, output_dir: Path, fmt: str) -> dict:
    expert_cols = [col for col in records.columns if col.startswith("expert_") and col.endswith("_prob")]
    num_experts = len(expert_cols)

    class_counts = records.groupby(["selected_expert", "label_name"]).size().rename("count").reset_index()
    class_counts["ratio"] = class_counts["count"] / class_counts.groupby("selected_expert")["count"].transform("sum")
    class_pivot = (
        class_counts.pivot(index="selected_expert", columns="label_name", values="ratio")
        .reindex(index=range(num_experts), columns=LABEL_NAMES)
        .fillna(0)
    )
    class_pivot.index = [f"Expert {i + 1}" for i in class_pivot.index]
    class_pivot.to_csv(output_dir / "expert_profile_class_distribution.csv")

    metrics = [
        "agreement",
        "full_correct",
        "full_confidence",
        "full_margin",
        "imu_confidence",
        "kp_confidence",
        "confidence_gap",
        "prob_js_divergence",
        "prob_l1_distance",
        "prob_cosine_distance",
        "feature_cosine_distance",
        "feature_l2_distance",
        "imu_feature_norm",
        "kp_feature_norm",
        "feature_norm_gap",
    ]
    profile = records.groupby("selected_expert")[metrics].mean().reset_index()
    profile["expert"] = profile["selected_expert"].map(lambda x: f"Expert {int(x) + 1}")
    profile.to_csv(output_dir / "expert_profile_metrics.csv", index=False)

    sns.set_theme(style="white", context="paper", font_scale=1.1)
    plt.figure(figsize=(11, 4.8))
    sns.heatmap(
        class_pivot,
        cmap=BLUE_GREEN_CMAP,
        annot=True,
        fmt=".2f",
        vmin=0,
        linewidths=0.6,
        linecolor="#E5EEF0",
        cbar_kws={"shrink": 0.82},
    )
    plt.xlabel("Class")
    plt.ylabel("Adaptive Expert")
    plt.title("Expert-Centric Class Profile")
    plt.tight_layout()
    plt.savefig(output_dir / f"fig_profile_expert_class_heatmap.{fmt}", dpi=300)
    plt.close()

    return {
        "class_distribution": class_pivot.reset_index().to_dict(orient="records"),
        "profile_metrics": profile.to_dict(orient="records"),
    }


FACTOR_LABELS = {
    "class": "True Class",
    "predicted_class": "Predicted Class",
    "unimodal_predictions": "Unimodal Predictions",
    "conflict": "Modality Conflict",
    "difficulty_confidence": "Prediction Confidence",
    "difficulty_margin": "Prediction Margin",
    "modality_confidence_gap": "Modality Confidence Gap",
    "prob_js_divergence": "Probability JS Divergence",
    "prob_l1_distance": "Probability L1 Distance",
    "feature_cosine_distance": "Feature Cosine Distance",
    "feature_l2_distance": "Feature L2 Distance",
    "feature_norm_gap": "Feature Norm Gap",
}


def eta_squared_from_groups(groups: list[np.ndarray]) -> float:
    groups = [g[~np.isnan(g)] for g in groups if len(g) > 0]
    if len(groups) < 2:
        return float("nan")
    all_values = np.concatenate(groups)
    if len(all_values) == 0:
        return float("nan")
    grand_mean = np.mean(all_values)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_total = np.sum((all_values - grand_mean) ** 2)
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def bin_series(values: pd.Series, bins: int = 3) -> pd.Series:
    unique_count = values.nunique(dropna=True)
    if unique_count <= bins:
        return values.astype(str)
    try:
        return pd.qcut(values, q=bins, labels=[f"Q{i + 1}" for i in range(bins)], duplicates="drop").astype(str)
    except ValueError:
        return pd.cut(values, bins=bins, labels=[f"B{i + 1}" for i in range(bins)], duplicates="drop").astype(str)


def effect_size_analysis(records: pd.DataFrame, output_dir: Path, fmt: str) -> pd.DataFrame:
    factor_specs = {
        "class": ("label_name", "categorical"),
        "predicted_class": ("full_pred", "categorical"),
        "unimodal_predictions": ("imu_pred", "categorical_pair"),
        "conflict": ("agreement", "categorical"),
        "difficulty_confidence": ("full_confidence", "numeric"),
        "difficulty_margin": ("full_margin", "numeric"),
        "modality_confidence_gap": ("confidence_gap", "numeric"),
        "prob_js_divergence": ("prob_js_divergence", "numeric"),
        "prob_l1_distance": ("prob_l1_distance", "numeric"),
        "feature_cosine_distance": ("feature_cosine_distance", "numeric"),
        "feature_l2_distance": ("feature_l2_distance", "numeric"),
        "feature_norm_gap": ("feature_norm_gap", "numeric"),
    }
    expert_cols = [col for col in records.columns if col.startswith("expert_") and col.endswith("_prob")]
    work = records.copy()
    work["unimodal_pair"] = work["imu_pred"].astype(str) + "-" + work["kp_pred"].astype(str)

    rows = []
    for factor_name, (col, kind) in factor_specs.items():
        if kind == "categorical_pair":
            factor_values = work["unimodal_pair"]
        elif kind == "numeric":
            factor_values = bin_series(work[col], bins=3)
        else:
            factor_values = work[col].astype(str)

        for expert_idx, expert_col in enumerate(expert_cols):
            groups = [work.loc[factor_values == level, expert_col].to_numpy(dtype=float) for level in pd.unique(factor_values)]
            eta2 = eta_squared_from_groups(groups)
            try:
                h_stat, p_value = stats.kruskal(*groups) if len(groups) >= 2 else (np.nan, np.nan)
            except ValueError:
                h_stat, p_value = np.nan, np.nan
            rows.append(
                {
                    "factor": factor_name,
                    "expert": f"Expert {expert_idx + 1}",
                    "eta_squared": eta2,
                    "kruskal_h": float(h_stat) if not np.isnan(h_stat) else np.nan,
                    "p_value": float(p_value) if not np.isnan(p_value) else np.nan,
                }
            )

    effect_df = pd.DataFrame(rows)
    effect_df.to_csv(output_dir / "router_factor_effect_sizes.csv", index=False)
    pivot = effect_df.pivot(index="expert", columns="factor", values="eta_squared")
    pivot = pivot.rename(columns=FACTOR_LABELS)

    sns.set_theme(style="white", context="paper", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(12, 5.6))
    sns.heatmap(
        pivot,
        cmap=BLUE_GREEN_CMAP,
        annot=True,
        fmt=".3f",
        ax=ax,
        linewidths=0.6,
        linecolor="#E5EEF0",
        cbar_kws={"shrink": 0.82},
    )
    plt.xlabel("Factor")
    plt.ylabel("Adaptive Expert Weight")
    plt.title("Effect Size of Factors on Router Weights")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.tight_layout(pad=0.8)
    plt.savefig(output_dir / f"fig_effect_size_router_weights.{fmt}", dpi=300)
    plt.close()

    factor_mean = effect_df.groupby("factor")["eta_squared"].mean().sort_values(ascending=False).reset_index()
    factor_mean.to_csv(output_dir / "router_factor_effect_size_ranking.csv", index=False)
    factor_mean_plot = factor_mean.copy()
    factor_mean_plot["factor_label"] = factor_mean_plot["factor"].map(FACTOR_LABELS)
    plt.figure(figsize=(11, 5.6))
    bar_palette = [BLUE_GREEN_CMAP(x) for x in np.linspace(0.92, 0.18, len(factor_mean_plot))]
    ax = sns.barplot(
        data=factor_mean_plot,
        x="factor_label",
        y="eta_squared",
        hue="factor_label",
        palette=dict(zip(factor_mean_plot["factor_label"], bar_palette)),
        legend=False,
        edgecolor="#334B53",
        linewidth=0.8,
    )
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Factor")
    plt.ylabel("Mean Eta Squared across Expert Weights")
    plt.title("Router Factor Effect Size Ranking")
    ax.grid(axis="y", color="#D8E3E6", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax, left=False, bottom=False)
    plt.tight_layout()
    plt.savefig(output_dir / f"fig_effect_size_factor_ranking.{fmt}", dpi=300)
    plt.close()
    return effect_df


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fix_random_seed(args.seed)

    record_path = args.output_dir / "adaptive_router_test_records.csv"
    if args.reuse_records and record_path.exists():
        config.logger.info("Reusing cached router records from %s", record_path)
        records = pd.read_csv(record_path)
    else:
        config.logger.info("Loading dataset from %s", args.dataset)
        _, _, test_set = load_saved_dataset(args.dataset)
        test_loader = build_loader(test_set, args.batch_size)
        config.logger.info("Loading PRISM and unimodal checkpoints from %s", args.checkpoint_dir)
        model = load_prism(args)
        records = collect_router_records(model, test_loader, args.device)
        records.to_csv(record_path, index=False)

    expert_profile = expert_profile_analysis(records, args.output_dir, args.figure_format)
    effect_sizes = effect_size_analysis(records, args.output_dir, args.figure_format)
    ranking = pd.read_csv(args.output_dir / "router_factor_effect_size_ranking.csv")

    summary = {
        "note": "Adaptive Router mechanism analysis based on test split router records. Probe predictability and counterfactual perturbation analyses have been removed to avoid redundant or confounded interpretation.",
        "test_n": int(len(records)),
        "expert_profile": expert_profile,
        "effect_size_ranking": ranking.to_dict(orient="records"),
        "effect_size_top": effect_sizes.sort_values("eta_squared", ascending=False).head(20).to_dict(orient="records"),
    }
    with open(args.output_dir / "adaptive_router_mechanism_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Adaptive Router Mechanism Analysis")
    print(f"Test records: {len(records)}")
    print("\nRouter factor effect size ranking:")
    print(ranking.to_string(index=False))
    print("\nTop router factor effect sizes:")
    print(effect_sizes.sort_values("eta_squared", ascending=False).head(20).to_string(index=False))
    print(f"\nSaved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
