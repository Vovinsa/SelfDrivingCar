import torch
from torch import nn

from .seresnet18 import make_seresnet18


class BranchedNetwork(nn.Module):
    def __init__(self, emb_size, num_commands, num_meas):
        super(BranchedNetwork, self).__init__()
        self.backbone = make_seresnet18(num_classes=emb_size)
        self.action_models = []
        for _ in range(num_commands):
            self.action_models.append(nn.Linear(emb_size * 2, 2))
        self.meas_embs = nn.Linear(num_meas, emb_size)

    def forward(self, img, measurements, command):
        img_embs = self.backbone(img)
        meas_embs = self.meas_embs(measurements)
        embs = torch.cat([img_embs, meas_embs], dim=1)
        preds = self.action_models[command](embs)
        angle = torch.sigmoid(preds[:, 0]) * 50
        speed = torch.tanh(preds[:, 1])
        return angle, speed
