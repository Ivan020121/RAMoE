from pathlib import Path
from config import config
from model.multimodal.vote import MultimodalVoteModel
from multimodal.train.base import get_unimodal_models
from util.torch_utils import (
    fix_random_seed,
    get_dataloaders,
    load_and_split_data,
    multi_evaluate,
)


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
    model_name = "Vote"

    logger.info(f"seed = {seed}")
    fix_random_seed(seed)
    train_set, val_set, test_set = load_and_split_data(
        Path("dataset"),
        imu_channel=imu_channels,
        load_dataset_path=Path("dataset/dataset_ag.pt"),
    )
    mode = "multimodal"
    imu_embedding_dim = 512
    kp_embedding_dim = 512

    for i in range(repeat_train):
        train_loader, val_loader, test_loader = get_dataloaders(
            train_set, val_set, test_set, batch_size=batch_size, data_type="both"
        )

        imu_encoder, kp_encoder = get_unimodal_models(
            train_set,
            val_set,
            test_set,
            device=device,
            pretrained_names=(
                "unimodal_imu 2026-01-31 17:49:32-latest.pt",
                "unimodal_kp 2026-01-25 13:00:01-latest.pt",
            ),
            with_head=True,
        )

        model = MultimodalVoteModel(imu_encoder, kp_encoder)

        multi_evaluate(
            model,
            model_name,
            test_loader,
            device=device,
            modal="multimodal",
        )
