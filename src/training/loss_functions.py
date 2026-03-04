import torch
from monai.losses.dice import DiceFocalLoss


class WeightedDiceFocalLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.class_weights = torch.tensor(
            [0.25, 0.25, 0.25, 0.25],  # eval others like [0.05, 0.3, 0.3, 0.35]
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
