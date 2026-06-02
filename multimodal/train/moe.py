from pathlib import Path
import torch
from torch import optim

from multimodal.train.base import Trainer, get_unimodal_models
from model.multimodal.moe import PRISM
from util.torch_utils import (
    evaluate_model,
    fix_random_seed,
    get_dataloaders,
    load_and_split_data,
    multi_evaluate,
)
from config import config

if __name__ == "__main__":
    logger = config.logger
    seed = 3407
    imu_channels = 9
    train_epoch = 40
    repeat_train = 1
    checkpoint_path = Path("checkpoint/use")
    device = "cuda:3"
    batch_size = 256
    lr = 1e-4
    model_name = "MoEAttn"

    logger.info("seed = %s", seed)
    fix_random_seed(seed, True)
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
                "augment_ratio": 0.5,
                "only_augmented": False,
            },
            "kp": {
                "scale": {"min": 0.5, "max": 1.5},
                "translation": {
                    "max": 0.5,
                },
                "augment_ratio": 0.5,
                "only_augmented": False,
            },
        },
    )
    modal = "multimodal"
    imu_embedding_dim = 512
    kp_embedding_dim = 256

    for i in range(repeat_train):
        imu_encoder, kp_encoder = get_unimodal_models(
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
        model = PRISM(imu_encoder, kp_encoder, 512, dropout=0.3)
        trainer = Trainer(
            model,
            "MOE",
            train_loader,
            val_loader,
            optimizer=optim.AdamW(model.parameters(), lr=lr),
            num_epochs=train_epoch,
            test_loader=test_loader,
            device=device,
            modal=modal,
            checkpoint_path=checkpoint_path,
            use_scheduler=True,
        )

        # trainer.train()

        model.load_state_dict(
            torch.load(
                checkpoint_path / "MOE 2026-01-15 17:58:42-latest.pt",
                map_location=device,
            )
        )

        multi_evaluate(
            model,
            f"{model_name}",
            test_loader,
            device=device,
            modal=modal,
            checkpoint_best_path=checkpoint_path
            / f"{trainer.model_weight_prefix}-best.pt",
        )
