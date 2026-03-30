from abc import ABC, abstractmethod
import torch


class BasePredictor(ABC):
    @abstractmethod
    def predict(self, imgs: torch.Tensor) -> torch.Tensor:
        """Predict class probabilities for the given images."""
        raise NotImplementedError("Subclasses must implement this method")
