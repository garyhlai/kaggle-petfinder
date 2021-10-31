import timm
import torch
import torch.nn as nn

class PawpularModel(nn.Module):
    def __init__(self, image_encoding_dim=128, feature_dim=12):
        super().__init__()
        self.model = timm.create_model("tf_efficientnet_b0_ns", pretrained=True, in_chans=3)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, image_encoding_dim)
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(image_encoding_dim+feature_dim, 1)
    
    def forward(self, image, features, targets=None):
        x = self.model(image)
        x = self.dropout(x)
        # (batch_size, 128, 1) -> (batch_size, 128+12, 1)
        x = torch.concat([x, features], dim=1)
        x = self.head(x)
        return x