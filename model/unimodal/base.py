from torch import nn


class UniModalityModel(nn.Module):
    def __init__(
        self, encoder, embedding_dim, num_classes: int = 10, with_feature=False
    ):
        super(UniModalityModel, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.with_feature = with_feature

    def forward(self, input_data):
        features = self.encoder(input_data)
        output = self.classifier(features)
        if self.with_feature:
            return features, output
        return output
