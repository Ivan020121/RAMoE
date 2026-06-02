from pathlib import Path
from torch import optim

from model.multimodal.moe_spare import PRISM
from multimodal.train.base import Trainer, get_unimodal_models
from util.torch_utils import (
    add_single_modal_samples,
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
    checkpoint_path = Path("checkpoint/multimodal")
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

    for i in range(repeat_train):
        imu_encoder, kp_encoder, history = get_unimodal_models(
            train_set,
            val_set,
            test_set,
            device=device,
            pretrained_names=(
                "unimodal_imu 2026-01-31 17:49:32-latest.pt",
                "unimodal_kp 2026-01-25 13:00:01-latest.pt",
            ),
            with_evaluate_rst=True,
            with_head=True,
            return_both=True,
        )

        train_loader, val_loader, test_loader = get_dataloaders(
            add_single_modal_samples(train_set),
            val_set,
            test_set,
            batch_size=batch_size,
            data_type="both",
        )
        model = PRISM(
            imu_encoder,
            kp_encoder,
            fusion_dim,
            dropout=0.3,
            expert_num=3,
            topk=1,
            history=history,
            device=device,
        )
        trainer = Trainer(
            model,
            "PRISM",
            train_loader,
            val_loader,
            optimizer=optim.AdamW(model.parameters(), lr=lr),
            num_epochs=train_epoch,
            test_loader=test_loader,
            device=device,
            modal=modal,
            checkpoint_path=checkpoint_path,
            use_scheduler=True,
            use_early_stopping=True,
            patience=10,
        )

        trainer.train()

        multi_evaluate(
            model,
            f"{model_name}",
            test_loader,
            device=device,
            modal=modal,
            checkpoint_best_path=checkpoint_path
            / f"{trainer.model_weight_prefix}-best.pt",
        )
