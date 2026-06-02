from pathlib import Path
import torch

from model.multimodal.moe import PRISM
from multimodal.train.base import get_unimodal_models
from util.torch_utils import (
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
    repeat_train = 1
    checkpoint_path = Path("checkpoint/multimodal")
    device = "cuda:0"
    batch_size = 256
    lr = 1e-4
    model_name = "PRISM"

    logger.info(f"seed = {seed}")
    fix_random_seed(seed, True)
    train_set, val_set, test_set = load_and_split_data(
        Path("dataset"),
        imu_channel=imu_channels,
        load_dataset_path=Path("dataset/dataset_ag.pt"),
        # save_generated_dataset=True,
        augment={
            "only_augment": False,
            "augment_ratio": 1,
            "imu": {
                "rotation": {"deg": 30},
                # "jitter": {"sigma": 0.05},
                "scale": {"min": 0.5, "max": 2},
                # "amplitude_warp": {"knot": 10, "sigma": 0.2},
                # "permutation": {"segments": 5},
                # "time_warp": {"factor": 1.2},
                "augment_ratio": 1,
                
            },
            "kp": {
                "scale": {"min": 0.5, "max": 2.5},
                "translation": {
                    "max": 1,
                },
                "augment_ratio": 1,
            },
        },
    )
    modal = "unimodal"
    imu_embedding_dim = 512
    kp_embedding_dim = 512
    print(len(train_set))

    for i in range(repeat_train):
        
        imu_encoder, kp_encoder = get_unimodal_models(
            train_set,
            val_set,
            test_set,
            pretrained_names=(
                "unimodal_imu 2026-01-31 17:49:32-latest.pt",
                "unimodal_kp 2026-01-25 13:00:01-latest.pt",
            ),
            with_head=True,
            return_both=True,
        )
        model = PRISM(imu_encoder, kp_encoder, 512, dropout=0.3)
        model.load_state_dict(torch.load(checkpoint_path / "PRISM 2026-02-01 22:21:27-latest.pt"))

        train_loader, val_loader, test_loader = get_dataloaders(
            train_set, val_set, test_set, batch_size=batch_size, data_type="both"
        )
        evaluate_model(
            model,
            "PRISM",
            test_loader,
            device=device,
            save_confusion_matrix=True,
            confusion_matrix_config={
                "row_normalize": True,
                "fmt": "svg"
            },
            modal="multimodal",
        )

        # train_loader, val_loader, test_loader = get_dataloaders(
        #     train_set, val_set, test_set, batch_size=batch_size, data_type="imu"
        # )
        # evaluate_model(
        #     imu_encoder,
        #     "imu_encoder-val",
        #     val_loader,
        #     device=device,
        #     # save_confusion_matrix=True,
        #     modal=modal,
        # )
        # evaluate_model(
        #     imu_encoder,
        #     "imu_encoder-test",
        #     test_loader,
        #     device=device,
        #     save_confusion_matrix=True,
        #     confusion_matrix_config={
        #         "row_normalize": True,
        #         "fmt": "svg"
        #     },
        #     modal=modal,
        # )

        # train_loader, val_loader, test_loader = get_dataloaders(
        #     train_set, val_set, test_set, batch_size=batch_size, data_type="kp"
        # )
        # evaluate_model(
        #     kp_encoder,
        #     "kp_encoder-val",
        #     val_loader,
        #     device=device,
        #     # save_confusion_matrix=True,
        #     modal=modal,
        # )
        # evaluate_model(
        #     kp_encoder,
        #     "kp_encoder-test",
        #     test_loader,
        #     device=device,
        #     save_confusion_matrix=True,
        #     confusion_matrix_config={
        #         "row_normalize": True,
        #         "fmt": "svg"
        #     },
        #     modal=modal,
        # )
        # if trainer.model_weight_prefix:
        #     model.load_state_dict(
        #         torch.load(checkpoint_path / f"{trainer.model_weight_prefix}-best.pt")
        #     )
        #     evaluate_model(
        #         model,
        #         f"{model_name}-best",
        #         test_loader,
        #         device=device,
        #         save_confusion_matrix=False,
        #         modal=modal,
        #     )
