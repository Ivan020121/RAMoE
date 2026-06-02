from pathlib import Path

from multimodal.train.base import get_unimodal_models
from util.torch_utils import (
    class_output,
    fix_random_seed,
    get_dataloaders,
    load_and_split_data,
    paried_test,
)
from config import config


if __name__ == "__main__":
    logger = config.logger
    seed = 3407
    imu_channels = 9
    train_epoch = 40
    repeat_train = 1
    checkpoint_path = Path("checkpoint/multimodal")
    device = "cuda:1"
    batch_size = 256
    lr = 1e-4
    model_name = "unimodal_compare"

    logger.info(f"seed = {seed}")
    fix_random_seed(seed, True)
    train_set, val_set, test_set = load_and_split_data(
        Path("dataset"),
        imu_channel=imu_channels,
        load_dataset_path=Path("dataset/dataset_ag.pt"),
    )
    modal = "unimodal"
    imu_embedding_dim = 512
    kp_embedding_dim = 512

    for i in range(repeat_train):
        imu_model, kp_model = get_unimodal_models(
            train_set,
            val_set,
            test_set,
            pretrained_names=(
                "unimodal_imu 2026-01-31 17:49:32-latest.pt",
                "unimodal_kp 2026-01-25 13:00:01-latest.pt",
            ),
            with_head=True,
        )

        _, _, test_loader = get_dataloaders(
            train_set, val_set, test_set, batch_size=batch_size, data_type="imu"
        )

        imu_ouput = class_output(
            imu_model,
            test_loader,
            device=device,
        )

        _, _, test_loader = get_dataloaders(
            train_set, val_set, test_set, batch_size=batch_size, data_type="kp"
        )

        kp_output = class_output(
            kp_model,
            test_loader,
            device=device,
        )

        statistics, pvals, qvals = paried_test(imu_ouput, kp_output)

        for i in range(10):
            print(
                f"Class {i}: statistic={statistics[i]:>8.2f}, p-value={pvals[i]:.4e}, q-value={qvals[i]:.4e}"
            )
