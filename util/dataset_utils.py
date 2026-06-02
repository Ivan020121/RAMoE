import random
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

from config import config

def load_and_split_data(user_ids, train_size=19, val_size=3, test_size=4, clean_kp: bool=False, imu_channels = 6):    
    """
    
    """
    random.shuffle(user_ids)
    
    train_users = user_ids[:train_size]
    val_users = user_ids[train_size:train_size+val_size]
    test_users = user_ids[train_size+val_size:train_size+val_size+test_size]
    
    config.logger.info(f"Train users: {sorted(train_users)}")
    config.logger.info(f"Val users: {sorted(val_users)}")
    config.logger.info(f"Test users: {sorted(test_users)}")
    
    def load_user_data(user_id: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        imu, kps, labels = torch.load(config.DATASET_PATH / f"user_{user_id}.pt")
        return imu[:, :, :imu_channels], kps, labels
    
    train_data = [load_user_data(uid) for uid in train_users]
    val_data = [load_user_data(uid) for uid in val_users]
    test_data = [load_user_data(uid) for uid in test_users]
    
    def merge_data(data_list: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        imus, kps, labels = zip(*data_list)
        merged_imus, merged_kps, merged_labels = torch.cat(imus), torch.cat(kps), torch.cat(labels)
        if clean_kp:
            mask = ~torch.isnan(merged_kps).any(dim=(1, 2))
            merged_imus, merged_kps, merged_labels = merged_imus[mask], merged_kps[mask], merged_labels[mask]
        # 随机索引打乱
        indices = torch.randperm(merged_imus.size(0))
        
        return merged_imus[indices], merged_kps[indices], merged_labels[indices]

    
    train_set = merge_data(train_data)
    val_set = merge_data(val_data)
    test_set = merge_data(test_data)

    config.logger.info(f"Train set length: {train_set[1].shape[0]}")      # (N_train, 200, 6)
    config.logger.info(f"Validation set length: {val_set[1].shape[0]}") # (N_train, 50, 35)
    config.logger.info(f"Teset set length: {test_set[1].shape[0]}")    # (N_train,)
    
    return train_set, val_set, test_set


def get_dataloaders(train_set, val_set, test_set, batch_size=32, data_type='imu'):
    X_imu_train, X_kp_train, y_train = train_set
    X_imu_val, X_kp_val, y_val = val_set
    X_imu_test, X_kp_test, y_test = test_set

    if data_type == 'imu':
        train_loader = DataLoader(TensorDataset(X_imu_train, y_train), batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(config.SEED))
        val_loader   = DataLoader(TensorDataset(X_imu_val, y_val),   batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(TensorDataset(X_imu_test, y_test), batch_size=batch_size, shuffle=False)
    elif data_type == 'kp':
        train_loader = DataLoader(TensorDataset(X_kp_train, y_train), batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(config.SEED))
        val_loader   = DataLoader(TensorDataset(X_kp_val, y_val),   batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(TensorDataset(X_kp_test, y_test), batch_size=batch_size, shuffle=False)

    config.logger.info(f"Train set size: {len(train_loader)}")
    config.logger.info(f"Validation set size: {len(val_loader)}")
    config.logger.info(f"Test set size: {len(test_loader)}")

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    load_and_split_data(1)