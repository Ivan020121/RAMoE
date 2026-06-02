import torch
import torch.nn as nn



class MultimodalVoteModel(nn.Module):
    def __init__(self, imu_classifier, kp_classifier):
        super(MultimodalVoteModel, self).__init__()
        self.imu_classifier = imu_classifier
        self.kp_classifier = kp_classifier

    def forward(self, imu_input, kp_input, return_features=False):
        imu_features = self.imu_classifier(imu_input)
        kp_features = self.kp_classifier(kp_input)
        # soft_voting_feat
        output = torch.mean(torch.stack([imu_features, kp_features]), dim=0)      

        if return_features:
            return output, imu_features, kp_features
        else:
            return output