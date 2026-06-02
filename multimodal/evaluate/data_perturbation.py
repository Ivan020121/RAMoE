from pathlib import Path
import torch

from model.multimodal.moe import PRISM
from multimodal.train.base import get_unimodal_models
from util.torch_utils import (
    augment_saved_dataset,
    evaluate_model,
    fix_random_seed,
    get_dataloaders,
    load_and_split_data,
)
from config import config

if __name__ == "__main__":
    logger = config.logger
    seed = 3407
    imu_channels = 9
    train_epoch = 40
    checkpoint_path = Path("checkpoint/use")
    device = "cuda:1"
    batch_size = 256
    lr = 6e-5
    model_name = "PRISM"

    logger.info("seed = %s", seed)
    fix_random_seed(seed, True)
    train_set, val_set, test_set = load_and_split_data(
        Path("dataset"),
        imu_channel=imu_channels,
        load_dataset_path=Path("dataset/dataset_ag.pt"),
    )

    modal = "multimodal"
    fusion_dim = 512

    imu_encoder, kp_encoder = get_unimodal_models(
        train_set,
        val_set,
        test_set,
        device=device,
        no_train=True,
        with_head=True,
        return_both=True,
    )

    train_loader, val_loader, test_loader = get_dataloaders(
        train_set,
        val_set,
        test_set,
        batch_size=batch_size,
        data_type="both",
    )
    model = PRISM(imu_encoder, kp_encoder, fusion_dim, dropout=0.3, device=device)
    model.load_state_dict(
        torch.load(
            checkpoint_path / "PRISM 2026-02-01 22:21:27-latest.pt",
            map_location=device,
        )
    )

    item = "translation"
    for i in range(11):
        print(f"{item}: {0.1 * i:.1f}")
        augment_test_set = augment_saved_dataset(
            test_set,
            {
                "augment_ratio": 1,
                "only_augment": True,
                "imu": {
                    # "rotation": {"deg": i * 10},
                    # "scale": {"min": 1 - 0.1 * i, "max": 1 + 0.1 * i},
                    # "jitter": {"sigma": -0.5 + 0.1 * i},
                    # "augment_ratio": 1,
                },
                "kp": {
                    # "scale": {"min": 1 - 0.1 * i, "max": 1 + 0.1 * i},
                    "translation": {"max": -0.5 + 0.1 * i},
                    "augment_ratio": 1,
                },
            },
        )

        _, _, test_loader = get_dataloaders(
            train_set,
            val_set,
            augment_test_set,
            batch_size=batch_size,
            data_type="both",
        )

        evaluate_model(
            model,
            f"{model_name}",
            test_loader,
            device=device,
            modal=modal,
        )
