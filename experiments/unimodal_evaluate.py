# Original RAMoE experiment; imports and checkpoint paths organized.
from pathlib import Path
from training.unimodal import get_unimodal_models
from utils.evaluation import class_output, paried_test
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
    device = 'cuda:1'
    batch_size = 256
    lr = 0.0001
    model_name = 'unimodal_compare'
    logger.info(f'seed = {seed}')
    fix_random_seed(seed, True)
    train_set, val_set, test_set = load_and_split_data(Path('dataset'), imu_channel=imu_channels, load_dataset_path=Path('dataset/dataset_ag.pt'))
    modal = 'unimodal'
    imu_embedding_dim = 512
    kp_embedding_dim = 512
    for i in range(repeat_train):
        imu_model, kp_model = get_unimodal_models(train_set, val_set, test_set, pretrained_names=('imu.pt', 'kp.pt'), with_head=True)
        _, _, test_loader = get_dataloaders(train_set, val_set, test_set, batch_size=batch_size, data_type='imu')
        imu_ouput = class_output(imu_model, test_loader, device=device)
        _, _, test_loader = get_dataloaders(train_set, val_set, test_set, batch_size=batch_size, data_type='kp')
        kp_output = class_output(kp_model, test_loader, device=device)
        statistics, pvals, qvals = paried_test(imu_ouput, kp_output)
        for i in range(10):
            print(f'Class {i}: statistic={statistics[i]:>8.2f}, p-value={pvals[i]:.4e}, q-value={qvals[i]:.4e}')
