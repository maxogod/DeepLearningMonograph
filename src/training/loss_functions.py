import torch
from monai.losses.dice import DiceFocalLoss


class WeightedDiceFocalLoss(torch.nn.Module):
    def __init__(self, weighted: bool):
        super().__init__()
        weights = [
            0.25,
            0.25,
            0.25,
            0.25,
        ]  # TODO: fine tune weights based on class imbalance
        if weighted:
            # weights = [0.02, 0.40, 0.23, 0.35] # based on class imbalance analysis
            weights = [0.02, 0.65, 0.15, 0.18]  # more aggressive weighting to boost NCR
        self.class_weights = torch.tensor(
            weights,
            dtype=torch.float32,
        )

        self.loss = DiceFocalLoss(
            include_background=True,
            to_onehot_y=False,
            softmax=True,
            weight=self.class_weights,
        )

    def forward(self, logits, targets):
        return self.loss(logits, targets)
