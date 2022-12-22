import torch
from torch import nn

from .seresnet18 import make_seresnet18


class BranchedNetwork(nn.Module):
    def __init__(self, emb_size, num_commands, num_meas):
        super(BranchedNetwork, self).__init__()

        self.hard_tanh = nn.Hardtanh(-1, 1)

        self.backbone = make_seresnet18(num_classes=emb_size)

        self.action_models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emb_size * 2, emb_size),
                nn.ReLU(),
                nn.Linear(emb_size, 2)
            )
            for _ in range(num_commands)
        ])

        self.meas_embs = nn.Linear(num_meas, emb_size)

    def forward(self, img, measurements, command):
        img_embs = self.backbone(img)
        meas_embs = self.meas_embs(measurements)

        embs = torch.cat([img_embs, meas_embs], dim=1)
        preds = torch.zeros(img.shape[0], 2)

        i = 0
        for emb, c in zip(embs, command):
            pred = self.action_models[c](emb)[0]
            preds[i] = pred
            i += 1

        angle = torch.sigmoid(preds[:, 0]) * 50
        speed = self.hard_tanh(preds[:, 1])
        return angle, speed
