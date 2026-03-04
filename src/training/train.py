from torch import amp, nn, optim
from torch.utils.data import DataLoader

from src.config.config import Config


class Trainer:
    def __init__(
        self,
        config: Config,
        model: nn.Module,
        criterion: nn.Module,
        lr: float,
        train: DataLoader,
        test: DataLoader | None = None,
    ):
        self.config = config

        self.model = model
        self.criterion = criterion
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scaler = amp.GradScaler("cuda")  # type: ignore

        self.train_loader = train
        self.test_loader = test

    def fit(self, num_epochs: int):
        pass


# probs = torch.softmax(outputs, dim=1)  # probabilities
# preds = torch.argmax(probs, dim=1)
# correct = (preds == masks).float()
# accuracy = correct.sum() / correct.numel()

# def compute_iou(preds, masks, num_classes):
#     ious = []

#     for cls in range(num_classes):
#         pred_cls = (preds == cls)
#         mask_cls = (masks == cls)

#         intersection = (pred_cls & mask_cls).sum().float()
#         union = (pred_cls | mask_cls).sum().float()

#         if union == 0:
#             continue

#         ious.append(intersection / union)

#     return torch.mean(torch.stack(ious))
# iou = compute_iou(preds, masks, num_classes=4)

# OR install torchmetrics
# from torchmetrics.classification import MulticlassAccuracy
# from torchmetrics.classification import MulticlassJaccardIndex
# accuracy_metric = MulticlassAccuracy(num_classes=4)
# iou_metric = MulticlassJaccardIndex(num_classes=4)
