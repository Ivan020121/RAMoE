from pathlib import Path
import numpy as np
from torch import optim
import torch

from model.multimodal.moe import PRISM
from multimodal.train.base import Trainer, get_unimodal_models
from util.torch_utils import (
    add_single_modal_samples,
    augment_saved_dataset,
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
    imu_channels = 3
    train_epoch = 40
    repeat_train = 1
    checkpoint_path = Path("checkpoint/multimodal")
    device = "cuda:0"
    batch_size = 256
    lr = 8e-5
    model_name = "PRISM"
    num_classes = 8

    logger.info("seed = %s", seed)
    fix_random_seed(seed, True)
    train_set, val_set, test_set = load_and_split_data(
        Path("new_dataset"),
        load_dataset_path=Path("new_dataset/dataset_unbalance_aug.pt"),
        # load_dataset_path=Path("new_dataset/newdataset_unbalance.pt"),
        imu_channel=imu_channels,
    )
    # train_set = augment_saved_dataset(
    #     train_set,
    #     augment={
    #         "imu": {
    #             "rotation": {"deg": 30},
    #             "scale": {"min": 0.5, "max": 2},
    #             "augment_ratio": 0.5,
    #             "only_augmented": False,
    #         },
    #         "kp": {
    #             "scale": {"min": 0.5, "max": 1.5},
    #             "translation": {
    #                 "max": 0.5,
    #             },
    #             "augment_ratio": 0.5,
    #             "only_augmented": False,
    #         },
    #         "augment_ratio": 1.0,
    #     },
    # )

    modal = "multimodal"
    fusion_dim = 512

    for i in range(repeat_train):
        imu_encoder, kp_encoder, history = get_unimodal_models(
            train_set,
            val_set,
            test_set,
            num_classes=num_classes,
            imu_epoch=25,
            kp_epoch=70,
            lr=1e-3,
            device=device,
            imu_channels=imu_channels,
            pretrained_names=(
                "newdata/unimodal_imu 2026-03-02 16:03:11-best.pt",
                "newdata/unimodal_kp 2026-03-02 16:03:16-latest.pt",
            ),
            with_head=True,
            return_both=True,
            with_evaluate_rst=True,
        )
        history = {
            "imu": np.array(
                [
                    0.84482759,
                    0.8852459,
                    0.85950413,
                    0.832,
                    0.70588235,
                    0.70149254,
                    0.71895425,
                    0.89830508,
                ]
            ),
            "kp": np.array(
                [0.992, 1.0, 1.0, 0.72727273, 0.85950413, 0.9047619, 0.90625, 0.8]
            ),
        }

        train_loader, val_loader, test_loader = get_dataloaders(
            train_set,
            val_set,
            test_set,
            batch_size=batch_size,
            data_type="both",
        )
        model = PRISM(
            imu_encoder,
            kp_encoder,
            fusion_dim,
            dropout=0.5,
            expert_num=3,
            topk=1,
            num_classes=num_classes,
            device=device,
            history=history,
        )
        # print(history)
        model.load_state_dict(torch.load("checkpoint/use/newdata/PRISM 2026-03-02 16:59:53-latest.pt", map_location=device))
        # trainer = Trainer(
        #     model,
        #     "PRISM",
        #     train_loader,
        #     val_loader,
        #     optimizer=optim.AdamW(model.parameters(), lr=lr),
        #     num_epochs=train_epoch,
        #     test_loader=test_loader,
        #     device=device,
        #     modal=modal,
        #     checkpoint_path=checkpoint_path,
        #     use_scheduler=False,
        #     use_early_stopping=True,
        #     patience=3,
        # )

        # trainer.train()

        evaluate_model(
            model,
            f"{model_name}",
            test_loader,
            device=device,
            modal=modal,
            labels=[
                "Drill",
                "Hammer",
                "Idle",
                "LiftBrick",
                "LiftRebar",
                "MeasureRebar",
                "TieRebar",
                "Travel",
            ],
            num_classes=num_classes,
            save_confusion_matrix=True,
            confusion_matrix_config={
                "fmt": "svg"
            },
            with_grouped_report=False,
        )
