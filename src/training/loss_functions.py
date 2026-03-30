import torch
from monai.losses.dice import DiceFocalLoss


class WeightedDiceFocalLoss(torch.nn.Module):
    def __init__(self, weighted: bool):
        super().__init__()
        weights = [0.25, 0.25, 0.25, 0.25]
        if weighted:
            weights = [0.02, 0.65, 0.15, 0.18]  # [0.02, 0.50, 0.25, 0.23]

        # Only foreground classes
        foreground_weights = weights[1:]  # [NCR, ED, ET]

        self.class_weights = torch.tensor(
            foreground_weights,
            dtype=torch.float32,
        )

        self.loss = DiceFocalLoss(
            include_background=False,
            to_onehot_y=False,
            softmax=True,
            weight=self.class_weights,
        )

    def forward(self, logits, targets):
        return self.loss(logits, targets)
