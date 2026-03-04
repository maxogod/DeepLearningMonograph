import numpy as np
import torch
import segmentation_models_3D as sm


class DiceFocalLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

        wt0, wt1, wt2, wt3 = 0.25, 0.25, 0.25, 0.25
        self.dice = sm.losses.DiceLoss(class_weights=np.array([wt0, wt1, wt2, wt3]))
        self.focal = sm.losses.CategoricalFocalLoss()

    def forward(self, y_pred, y_true):  # TODO: maybe weight the losses
        return self.dice(y_pred, y_true) + self.focal(y_pred, y_true)
