import torch
import torch.nn as nn


class BaselineModel(nn.Module):
    def __init__(
        self,
        imu_encoder,
        kp_encoder,
        imu_embedding_dim,
        kp_embedding_dim,
        num_classes: int = 10,
    ):
        super(BaselineModel, self).__init__()
        self.imu_encoder = imu_encoder
        self.kp_encoder = kp_encoder
        self.classifier = nn.Linear(imu_embedding_dim + kp_embedding_dim, num_classes)

    def forward(self, imu_input, kp_input, return_features=False):
        imu_features = self.imu_encoder(imu_input)
        kp_features = self.kp_encoder(kp_input)

        conbined_feat = torch.cat((imu_features, kp_features), dim=1)

        output = self.classifier(conbined_feat)
        if return_features:
            return output, imu_features, kp_features
        else:
            return output
