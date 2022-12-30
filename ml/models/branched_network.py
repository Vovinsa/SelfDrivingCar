import torch
from torch import nn

from .seresnet18 import make_seresnet18


class BranchedNetwork(nn.Module):
    def __init__(self, emb_size):
        super(BranchedNetwork, self).__init__()
        self.hard_tanh = nn.Hardtanh(-1, 1)
        self.backbone = make_seresnet18(num_classes=emb_size)
        self.action_models = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, 2)
            )

    def forward(self, img):
        img_embs = self.backbone(img)

        preds = self.action_models(img_embs)

        angle = torch.sigmoid(preds[:, 0]) * 50
        speed = self.hard_tanh(preds[:, 1])
        return angle, speed
