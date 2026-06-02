from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import config
from model.multimodal.moe import PRISM
from model.unimodal.base import UniModalityModel
from model.unimodal.imutconv import TemporalConvEncoder
from model.unimodal.stgcn import STGCN_Encoder


TARGETS = {
    "adaptive_expert": "Adaptive Expert",
    "prior_modality": "Prior Modality",
}


FEATURE_GROUPS = {
    "adaptive_expert": {
        "majority_baseline": {"categorical": [], "numeric": []},
        "prior_side": {
            "categorical": ["imu_pred", "kp_pred", "agreement", "prior_selected_modality"],
            "numeric": [
                "imu_confidence",
                "kp_confidence",
                "confidence_gap",
                "prob_js_divergence",
                "prob_l1_distance",
                "prob_cosine_distance",
                "prior_weight_imu",
                "prior_weight_kp",
                "prior_entropy",
                "history_f1_imu",
                "history_f1_kp",
            ],
        },
        "adaptive_feature_side": {
            "categorical": [],
            "numeric": [
                "feature_cosine_distance",
                "feature_l2_distance",
                "imu_feature_norm",
                "kp_feature_norm",
                "feature_norm_gap",
            ],
        },
        "combined": {
            "categorical": ["imu_pred", "kp_pred", "agreement", "prior_selected_modality"],
            "numeric": [
                "imu_confidence",
                "kp_confidence",
                "confidence_gap",
                "prob_js_divergence",
                "prob_l1_distance",
                "prob_cosine_distance",
                "prior_weight_imu",
                "prior_weight_kp",
                "prior_entropy",
                "history_f1_imu",
                "history_f1_kp",
                "feature_cosine_distance",
                "feature_l2_distance",
                "imu_feature_norm",
                "kp_feature_norm",
                "feature_norm_gap",
            ],
        },
    },
    "prior_modality": {
        "majority_baseline": {"categorical": [], "numeric": []},
        "adaptive_side": {
            "categorical": ["adaptive_selected_expert"],
            "numeric": [
                "adaptive_weight_1",
                "adaptive_weight_2",
                "adaptive_weight_3",
                "adaptive_weight_4",
                "adaptive_entropy",
                "adaptive_confidence",
                "feature_cosine_distance",
                "feature_l2_distance",
                "imu_feature_norm",
                "kp_feature_norm",
                "feature_norm_gap",
            ],
        },
        "prior_visible_side": {
            "categorical": ["imu_pred", "kp_pred", "agreement"],
            "numeric": [
                "imu_confidence",
                "kp_confidence",
                "confidence_gap",
                "prob_js_divergence",
                "prob_l1_distance",
                "prob_cosine_distance",
                "history_f1_imu",
                "history_f1_kp",
            ],
        },
        "combined": {
            "categorical": ["adaptive_selected_expert", "imu_pred", "kp_pred", "agreement"],
            "numeric": [
                "adaptive_weight_1",
                "adaptive_weight_2",
                "adaptive_weight_3",
                "adaptive_weight_4",
                "adaptive_entropy",
                "adaptive_confidence",
                "feature_cosine_distance",
                "feature_l2_distance",
                "imu_feature_norm",
                "kp_feature_norm",
                "feature_norm_gap",
                "imu_confidence",
                "kp_confidence",
                "confidence_gap",
                "prob_js_divergence",
                "prob_l1_distance",
                "prob_cosine_distance",
                "history_f1_imu",
                "history_f1_kp",
            ],
        },
    },
}


CLASSIFIER_BUILDERS = {
    "DummyMostFrequent": lambda seed: DummyClassifier(strategy="most_frequent"),
    "LogisticRegression": lambda seed: LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        multi_class="auto",
    ),
    "LinearSVC": lambda seed: LinearSVC(class_weight="balanced", dual=False, max_iter=10000, random_state=seed),
    "RandomForest": lambda seed: RandomForestClassifier(
        n_estimators=120,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    ),
}

if XGBClassifier is not None:
    CLASSIFIER_BUILDERS["XGBoost"] = lambda seed: XGBClassifier(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softmax",
        eval_metric="mlogloss",
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )


def build_classifier(model_name: str, seed: int, n_classes: int):
    if model_name != "XGBoost":
        return CLASSIFIER_BUILDERS[model_name](seed)
    if XGBClassifier is None:
        raise RuntimeError("XGBoost is requested but xgboost is not installed.")
    common = dict(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )
    if n_classes <= 2:
        return XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common)
    return XGBClassifier(objective="multi:softmax", eval_metric="mlogloss", num_class=n_classes, **common)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bidirectional explainability between Prior MoE and Adaptive MoE.")
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
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "img" / "moe_mutual_explainability")
    parser.add_argument("--figure-format", choices=["png", "svg", "pdf"], default="svg")
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
    imu_model = UniModalityModel(imu_encoder, args.fusion_dim, num_classes=args.num_classes, with_feature=True)
    kp_model = UniModalityModel(kp_encoder, args.fusion_dim, num_classes=args.num_classes, with_feature=True)
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


def probability_js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    m = 0.5 * (p + q)
    kl_pm = (p * (torch.log(p.clamp_min(1e-12)) - torch.log(m.clamp_min(1e-12)))).sum(dim=1)
    kl_qm = (q * (torch.log(q.clamp_min(1e-12)) - torch.log(m.clamp_min(1e-12)))).sum(dim=1)
    return 0.5 * (kl_pm + kl_qm)


def entropy(prob: torch.Tensor) -> torch.Tensor:
    return -(prob * torch.log(prob.clamp_min(1e-12))).sum(dim=1)


@torch.no_grad()
def collect_records(model: PRISM, loader, device: str, split: str) -> pd.DataFrame:
    rows = []
    sample_offset = 0
    history_imu = torch.tensor(model.history["imu"], dtype=torch.float32, device=device)
    history_kp = torch.tensor(model.history["kp"], dtype=torch.float32, device=device)
    for x, y in tqdm(loader, desc=f"Collecting {split} mutual explainability records"):
        imu = x["imu"].to(device).float()
        kp = x["kp"].to(device).float()
        labels = y.to(device).long()

        imu_features, imu_logits = model.imu_encoder(imu)
        kp_features, kp_logits = model.kp_encoder(kp)
        imu_prob = F.softmax(imu_logits, dim=1)
        kp_prob = F.softmax(kp_logits, dim=1)

        imu_conf, imu_pred = imu_prob.max(dim=1)
        kp_conf, kp_pred = kp_prob.max(dim=1)
        agreement = imu_pred == kp_pred

        rel_imu = history_imu[imu_pred]
        rel_kp = history_kp[kp_pred]
        prior_scores = torch.stack([imu_conf * rel_imu, kp_conf * rel_kp], dim=1)
        prior_weights = F.softmax(prior_scores, dim=1)
        prior_selected = prior_weights.argmax(dim=1)

        feat = torch.cat([imu_features, kp_features], dim=1)
        adaptive_prob = F.softmax(model.router(feat), dim=1)
        adaptive_selected = adaptive_prob.argmax(dim=1)

        prob_js = probability_js_divergence(imu_prob, kp_prob)
        prob_l1 = torch.abs(imu_prob - kp_prob).sum(dim=1)
        prob_cosine_distance = 1 - F.cosine_similarity(imu_prob, kp_prob, dim=1)

        imu_features_normed = F.normalize(imu_features, dim=1)
        kp_features_normed = F.normalize(kp_features, dim=1)
        feature_cosine_distance = 1 - F.cosine_similarity(imu_features_normed, kp_features_normed, dim=1)
        feature_l2_distance = torch.linalg.vector_norm(imu_features_normed - kp_features_normed, ord=2, dim=1)
        imu_feature_norm = torch.linalg.vector_norm(imu_features, ord=2, dim=1)
        kp_feature_norm = torch.linalg.vector_norm(kp_features, ord=2, dim=1)

        for i in range(labels.size(0)):
            row = {
                "split": split,
                "sample_id": sample_offset + i,
                "label": int(labels[i].item()),
                "imu_pred": int(imu_pred[i].item()),
                "kp_pred": int(kp_pred[i].item()),
                "agreement": int(agreement[i].item()),
                "imu_confidence": float(imu_conf[i].item()),
                "kp_confidence": float(kp_conf[i].item()),
                "confidence_gap": float(torch.abs(imu_conf[i] - kp_conf[i]).item()),
                "prob_js_divergence": float(prob_js[i].item()),
                "prob_l1_distance": float(prob_l1[i].item()),
                "prob_cosine_distance": float(prob_cosine_distance[i].item()),
                "history_f1_imu": float(rel_imu[i].item()),
                "history_f1_kp": float(rel_kp[i].item()),
                "prior_weight_imu": float(prior_weights[i, 0].item()),
                "prior_weight_kp": float(prior_weights[i, 1].item()),
                "prior_entropy": float(entropy(prior_weights)[i].item()),
                "prior_selected_modality": int(prior_selected[i].item()),
                "feature_cosine_distance": float(feature_cosine_distance[i].item()),
                "feature_l2_distance": float(feature_l2_distance[i].item()),
                "imu_feature_norm": float(imu_feature_norm[i].item()),
                "kp_feature_norm": float(kp_feature_norm[i].item()),
                "feature_norm_gap": float(torch.abs(imu_feature_norm[i] - kp_feature_norm[i]).item()),
                "adaptive_selected_expert": int(adaptive_selected[i].item()),
                "adaptive_entropy": float(entropy(adaptive_prob)[i].item()),
                "adaptive_confidence": float(adaptive_prob[i].max().item()),
            }
            for expert_idx in range(adaptive_prob.size(1)):
                row[f"adaptive_weight_{expert_idx + 1}"] = float(adaptive_prob[i, expert_idx].item())
            rows.append(row)
        sample_offset += labels.size(0)
    return pd.DataFrame(rows)


def make_preprocessor(categorical_cols: list[str], numeric_cols: list[str]) -> ColumnTransformer:
    transformers = []
    if categorical_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols))
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if not transformers:
        transformers.append(("dummy", StandardScaler(), ["constant_feature"]))
    return ColumnTransformer(transformers=transformers)


def metric_dict(y_true: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
    }


def build_pipeline(target: str, group: str, model_name: str, seed: int, n_classes: int):
    spec = FEATURE_GROUPS[target][group]
    classifier = build_classifier(model_name, seed, n_classes)
    pipeline = Pipeline(
        steps=[
            ("preprocess", make_preprocessor(spec["categorical"], spec["numeric"])),
            ("classifier", classifier),
        ]
    )
    feature_cols = spec["categorical"] + spec["numeric"]
    if not feature_cols:
        feature_cols = ["constant_feature"]
    return pipeline, feature_cols


def run_probe(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, seed: int):
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    for df in [train_df, val_df, test_df]:
        df["constant_feature"] = 0.0

    val_rows = []
    test_rows = []
    selected_rows = []
    target_cols = {
        "adaptive_expert": "adaptive_selected_expert",
        "prior_modality": "prior_selected_modality",
    }
    for target, target_col in target_cols.items():
        y_train = train_df[target_col].to_numpy()
        y_val = val_df[target_col].to_numpy()
        y_test = test_df[target_col].to_numpy()
        for group in FEATURE_GROUPS[target]:
            model_names = ["DummyMostFrequent"] if group == "majority_baseline" else [
                name for name in CLASSIFIER_BUILDERS if name != "DummyMostFrequent"
            ]
            for model_name in model_names:
                pipeline, feature_cols = build_pipeline(target, group, model_name, seed, len(np.unique(y_train)))
                pipeline.fit(train_df[feature_cols], y_train)
                val_pred = pipeline.predict(val_df[feature_cols])
                test_pred = pipeline.predict(test_df[feature_cols])
                val_rows.append(
                    {
                        "target": target,
                        "target_name": TARGETS[target],
                        "feature_group": group,
                        "model": model_name,
                        "train_n": len(train_df),
                        "val_n": len(val_df),
                        **metric_dict(y_val, val_pred),
                    }
                )
                test_rows.append(
                    {
                        "target": target,
                        "target_name": TARGETS[target],
                        "feature_group": group,
                        "model": model_name,
                        "train_n": len(train_df),
                        "test_n": len(test_df),
                        **metric_dict(y_test, test_pred),
                    }
                )

    val_results = pd.DataFrame(val_rows)
    test_results = pd.DataFrame(test_rows)
    for (target, group), subset in val_results.groupby(["target", "feature_group"]):
        best = subset.sort_values("macro_f1", ascending=False).iloc[0]
        test_match = test_results[
            (test_results["target"] == target)
            & (test_results["feature_group"] == group)
            & (test_results["model"] == best["model"])
        ].iloc[0]
        row = test_match.to_dict()
        row["selected_by_val_model"] = best["model"]
        row["val_macro_f1"] = float(best["macro_f1"])
        row["val_accuracy"] = float(best["accuracy"])
        selected_rows.append(row)
    selected = pd.DataFrame(selected_rows).sort_values(["target", "macro_f1"], ascending=[True, False])
    return val_results, test_results, selected


GROUP_LABELS = {
    "majority_baseline": "Majority Baseline",
    "prior_side": "Prior-side Only",
    "adaptive_feature_side": "Adaptive Feature-side Only",
    "adaptive_side": "Adaptive-side Only",
    "prior_visible_side": "Prior-visible Side Only",
    "combined": "Combined",
}

GROUP_COLORS = {
    "Majority Baseline": "#BAD2E1",
    "Prior-side Only": "#9ED17B",
    "Adaptive Feature-side Only": "#9DC7DD",
    "Adaptive-side Only": "#58B8D1",
    "Prior-visible Side Only": "#3D9F3C",
    "Combined": "#04579B",
}


def plot_mutual_explainability(selected: pd.DataFrame, output_dir: Path, fmt: str) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    plot_df = selected.copy()
    plot_df["feature_group_label"] = plot_df["feature_group"].map(GROUP_LABELS)
    order = ["Majority Baseline", "Prior-side Only", "Adaptive Feature-side Only", "Adaptive-side Only", "Prior-visible Side Only", "Combined"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, target in zip(axes, ["adaptive_expert", "prior_modality"]):
        subset = plot_df[plot_df["target"] == target].copy()
        subset = subset[subset["feature_group_label"].notna()]
        target_order = [x for x in order if x in subset["feature_group_label"].values]
        sns.barplot(
            data=subset,
            x="feature_group_label",
            y="macro_f1",
            hue="feature_group_label",
            palette=GROUP_COLORS,
            hue_order=target_order,
            legend=False,
            edgecolor="#334B53",
            linewidth=0.8,
            ax=ax,
            order=target_order,
        )
        ax.set_title(f"Target: {TARGETS[target]}")
        ax.set_xlabel("")
        ax.set_ylabel("Test Macro-F1" if ax is axes[0] else "")
        ax.grid(axis="y", color="#D8E3E6", linewidth=0.8)
        ax.grid(axis="x", visible=False)
        sns.despine(ax=ax, left=False, bottom=False)
        ax.tick_params(axis="x", rotation=35)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")
    plt.tight_layout()
    plt.savefig(output_dir / f"fig_mutual_explainability.{fmt}", dpi=300)
    plt.close()

    target_to_file = {
        "adaptive_expert": "fig_prior_explains_adaptive",
        "prior_modality": "fig_adaptive_explains_prior",
    }
    for target, filename in target_to_file.items():
        subset = plot_df[plot_df["target"] == target].copy()
        subset = subset[subset["feature_group_label"].notna()]
        target_order = [x for x in order if x in subset["feature_group_label"].values]
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        sns.barplot(
            data=subset,
            x="feature_group_label",
            y="macro_f1",
            hue="feature_group_label",
            palette=GROUP_COLORS,
            hue_order=target_order,
            legend=False,
            edgecolor="#334B53",
            linewidth=0.8,
            ax=ax,
            order=target_order,
        )
        ax.set_ylim(0, 1.0)
        ax.set_title(f"Target: {TARGETS[target]}")
        ax.set_xlabel("")
        ax.set_ylabel("Test Macro-F1")
        ax.grid(axis="y", color="#D8E3E6", linewidth=0.8)
        ax.grid(axis="x", visible=False)
        sns.despine(ax=ax, left=False, bottom=False)
        ax.tick_params(axis="x", rotation=25)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", padding=3, fontsize=9)
        plt.tight_layout()
        plt.savefig(output_dir / f"{filename}.{fmt}", dpi=300)
        plt.close()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fix_random_seed(args.seed)

    record_paths = {
        "train": args.output_dir / "mutual_explainability_records_train.csv",
        "val": args.output_dir / "mutual_explainability_records_val.csv",
        "test": args.output_dir / "mutual_explainability_records_test.csv",
    }
    if args.reuse_records and all(path.exists() for path in record_paths.values()):
        train_records = pd.read_csv(record_paths["train"])
        val_records = pd.read_csv(record_paths["val"])
        test_records = pd.read_csv(record_paths["test"])
    else:
        train_set, val_set, test_set = load_saved_dataset(args.dataset)
        model = load_prism(args)
        train_records = collect_records(model, build_loader(train_set, args.batch_size), args.device, "train")
        val_records = collect_records(model, build_loader(val_set, args.batch_size), args.device, "val")
        test_records = collect_records(model, build_loader(test_set, args.batch_size), args.device, "test")
        train_records.to_csv(record_paths["train"], index=False)
        val_records.to_csv(record_paths["val"], index=False)
        test_records.to_csv(record_paths["test"], index=False)

    val_results, test_results, selected = run_probe(train_records, val_records, test_records, args.seed)
    val_results.to_csv(args.output_dir / "mutual_explainability_validation_results.csv", index=False)
    test_results.to_csv(args.output_dir / "mutual_explainability_test_results.csv", index=False)
    selected.to_csv(args.output_dir / "mutual_explainability_selected_test_results.csv", index=False)
    plot_mutual_explainability(selected, args.output_dir, args.figure_format)

    summary = {
        "note": "Bidirectional probe analysis. Models are trained on train split, selected on validation split, and reported on test split.",
        "train_n": int(len(train_records)),
        "val_n": int(len(val_records)),
        "test_n": int(len(test_records)),
        "selected_test_results": selected.to_dict(orient="records"),
    }
    with open(args.output_dir / "mutual_explainability_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("MoE Mutual Explainability Analysis")
    print(f"Train records: {len(train_records)}")
    print(f"Val records: {len(val_records)}")
    print(f"Test records: {len(test_records)}")
    print("\nVal-selected test results:")
    print(selected.to_string(index=False))
    print(f"\nSaved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
