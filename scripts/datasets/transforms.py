import numpy as np
import torch
from torch import Tensor


class MinMaxNormalize:
    
    def __init__(self, min: np.ndarray, max: np.ndarray) -> None:
        self.min = min
        self.max = max
        
    def __call__(self, image: np.ndarray) -> np.ndarray:
        log_image = np.log(np.abs(image) + np.spacing(1))
        normalized_image = (log_image - self.min) / (self.max - self.min)
        normalized_image = np.clip(normalized_image, 0, 1)
        return normalized_image
    

class MinMaxDenormalize:

    def __init__(self, min: np.ndarray | Tensor, max: np.ndarray | Tensor) -> None:
        self.min = min
        self.max = max
        
    def denorm_ndarray(self, norm_image: np.ndarray) -> np.ndarray:
        log_image = (self.max - self.min) * norm_image.astype(np.float32) + self.min

        return np.exp(log_image) - np.spacing(1)
    
    def denorm_tensor(self, norm_image: Tensor) -> Tensor:
        log_image = (self.max - self.min) * norm_image.to(torch.float32) + self.min

        return torch.exp(log_image) - np.spacing(1)

    def __call__(self, norm_image: np.ndarray | Tensor) -> np.ndarray | Tensor:
        if isinstance(norm_image, np.ndarray):
            return self.denorm_ndarray(norm_image)
        elif isinstance(norm_image, Tensor):
            return self.denorm_tensor(norm_image)