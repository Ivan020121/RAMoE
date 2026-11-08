# Original RAMoE experiment; imports and checkpoint paths organized.
from pathlib import Path
import torch
from models.ramoe import PRISM
from training.unimodal import get_unimodal_models
from utils.data import augment_saved_dataset, get_dataloaders, load_and_split_data
from utils.evaluation import evaluate_model
from utils.reproducibility import fix_random_seed
from config import config
if __name__ == '__main__':
    logger = config.logger
    seed = 3407
    imu_channels = 9
    train_epoch = 40
    checkpoint_path = Path('checkpoints/ramoe')
    device = 'cuda:1'
    batch_size = 256
    lr = 6e-05
    model_name = 'PRISM'
    logger.info('seed = %s', seed)
    fix_random_seed(seed, True)
    train_set, val_set, test_set = load_and_split_data(Path('dataset'), imu_channel=imu_channels, load_dataset_path=Path('dataset/dataset_ag.pt'))
    modal = 'multimodal'
    fusion_dim = 512
    imu_encoder, kp_encoder = get_unimodal_models(train_set, val_set, test_set, device=device, no_train=True, with_head=True, return_both=True)
    train_loader, val_loader, test_loader = get_dataloaders(train_set, val_set, test_set, batch_size=batch_size, data_type='both')
    model = PRISM(imu_encoder, kp_encoder, fusion_dim, dropout=0.3, device=device)
    model.load_state_dict(torch.load(checkpoint_path / 'latest.pt', map_location=device))
    item = 'translation'
    for i in range(11):
        print(f'{item}: {0.1 * i:.1f}')
        augment_test_set = augment_saved_dataset(test_set, {'augment_ratio': 1, 'only_augment': True, 'imu': {}, 'kp': {'translation': {'max': -0.5 + 0.1 * i}, 'augment_ratio': 1}})
        _, _, test_loader = get_dataloaders(train_set, val_set, augment_test_set, batch_size=batch_size, data_type='both')
        evaluate_model(model, f'{model_name}', test_loader, device=device, modal=modal)
