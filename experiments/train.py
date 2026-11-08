# Original RAMoE experiment; imports and checkpoint paths organized.
from pathlib import Path
from torch import optim
from models.ramoe import PRISM
from training.trainer import Trainer
from training.unimodal import get_unimodal_models
from utils.data import add_single_modal_samples, get_dataloaders, load_and_split_data
from utils.evaluation import evaluate_model, multi_evaluate
from utils.reproducibility import fix_random_seed
from config import config
if __name__ == '__main__':
    logger = config.logger
    seed = 3407
    imu_channels = 9
    train_epoch = 40
    repeat_train = 1
    checkpoint_path = Path('checkpoints/ramoe')
    device = 'cuda:0'
    batch_size = 256
    lr = 6e-05
    model_name = 'PRISM'
    logger.info('seed = %s', seed)
    fix_random_seed(seed, True)
    train_set, val_set, test_set = load_and_split_data(Path('dataset'), imu_channel=imu_channels, load_dataset_path=Path('dataset/dataset_ag.pt'))
    modal = 'multimodal'
    fusion_dim = 512
    for i in range(repeat_train):
        imu_encoder, kp_encoder = get_unimodal_models(train_set, val_set, test_set, device=device, pretrained_names=('imu.pt', 'kp.pt'), with_head=True, return_both=True)
        train_loader, val_loader, test_loader = get_dataloaders(add_single_modal_samples(train_set), val_set, test_set, batch_size=batch_size, data_type='both')
        model = PRISM(imu_encoder, kp_encoder, fusion_dim, dropout=0.3, device=device)
        trainer = Trainer(model, 'PRISM', train_loader, val_loader, optimizer=optim.AdamW(model.parameters(), lr=lr), num_epochs=train_epoch, test_loader=test_loader, device=device, modal=modal, checkpoint_path=checkpoint_path, use_scheduler=False, use_early_stopping=True, patience=6)
        trainer.train()
        multi_evaluate(model, f'{model_name}', test_loader, device=device, modal=modal, checkpoint_best_path=checkpoint_path / f'{trainer.model_weight_prefix}-best.pt')
