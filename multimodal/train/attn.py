from pathlib import Path
import torch
from torch import optim

from config import config
from model.multimodal.attn import MultiModalAttn
from multimodal.train.base import Trainer, get_unimodal_models
from util.torch_utils import evaluate_model, fix_random_seed, get_dataloaders, load_and_split_data, multi_evaluate


if __name__ == "__main__":
    logger = config.logger
    seed = 3407
    imu_channels = 9 
    train_epoch = 50
    repeat_train = 1
    checkpoint_path = Path("checkpoint/multimodal")
    device = "cuda:0"
    batch_size = 256
    lr = 1e-5
    model_name = "Attn"

    logger.info(f"seed = {seed}")
    fix_random_seed(seed, True)
    train_set, val_set, test_set = load_and_split_data(
        Path("dataset"),
        imu_channel=imu_channels,
        load_dataset_path=Path("dataset/dataset.pt"),
        augment={
            "imu": {
                "rotation": {"deg": 30}, 
                "scale": {"min": 0.5, "max": 2},
                "augment_ratio": 0, 
                "only_augmented": False
            },
            "kp": {
                "scale": {
                    "min": 0.5, 
                    "max": 1.5
                },
                "translation": {
                    "max": 0.5, 
                },
                "augment_ratio": 1, 
                "only_augmented": False
            }
        }
    )
    modal = "multimodal"
    imu_embedding_dim = 512
    kp_embedding_dim = 256

    for i in range(repeat_train):
        train_loader, val_loader, test_loader = get_dataloaders(
            train_set, val_set, test_set, batch_size=batch_size, data_type="both"
        )

        imu_pretrained, kp_pretrained = get_unimodal_models(
            train_set,
            val_set,
            test_set,
            device=device,
            pretrained_names=(
                "unimodal_imu 2026-01-08 15:58:38-latest.pt",
                "unimodal_kp 2026-01-06 21:00:06-latest.pt",
            ),
        )

        model = MultiModalAttn(imu_pretrained, kp_pretrained)
        trainer = Trainer(model,
                            "Attn",
                            train_loader,
                            val_loader,
                            optimizer=optim.AdamW(model.parameters(), lr=lr),
                            num_epochs = train_epoch,
                            test_loader = test_loader,
                            device = device,
                            modal = modal,
                            checkpoint_path = checkpoint_path,
                            use_scheduler = True)
        
        
        trainer.train()
        
        multi_evaluate(
            model,
            model_name,
            test_loader,
            device=device,
            modal="multimodal",
            checkpoint_best_path=checkpoint_path / f"{trainer.model_weight_prefix}-best.pt",
        )
