# Original RAMoE experiment; imports and checkpoint paths organized.
from pathlib import Path
import torch
from models.ramoe import PRISM
from training.unimodal import get_unimodal_models
from utils.evaluation import evaluate_model
from utils.reproducibility import fix_random_seed
from utils.data import get_dataloaders, load_and_split_data
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
    lr = 0.0001
    model_name = 'PRISM'
    logger.info(f'seed = {seed}')
    fix_random_seed(seed, True)
    train_set, val_set, test_set = load_and_split_data(Path('dataset'), imu_channel=imu_channels, load_dataset_path=Path('dataset/dataset_ag.pt'), augment={'only_augment': False, 'augment_ratio': 1, 'imu': {'rotation': {'deg': 30}, 'scale': {'min': 0.5, 'max': 2}, 'augment_ratio': 1}, 'kp': {'scale': {'min': 0.5, 'max': 2.5}, 'translation': {'max': 1}, 'augment_ratio': 1}})
    modal = 'unimodal'
    imu_embedding_dim = 512
    kp_embedding_dim = 512
    print(len(train_set))
    for i in range(repeat_train):
        imu_encoder, kp_encoder = get_unimodal_models(train_set, val_set, test_set, pretrained_names=('imu.pt', 'kp.pt'), with_head=True, return_both=True)
        model = PRISM(imu_encoder, kp_encoder, 512, dropout=0.3)
        model.load_state_dict(torch.load(checkpoint_path / 'latest.pt', map_location='cpu'))
        train_loader, val_loader, test_loader = get_dataloaders(train_set, val_set, test_set, batch_size=batch_size, data_type='both')
        Path('img').mkdir(parents=True, exist_ok=True)
        evaluate_model(model, 'PRISM', test_loader, device=device, save_confusion_matrix=True, confusion_matrix_config={'row_normalize': True, 'fmt': 'svg'}, modal='multimodal')
