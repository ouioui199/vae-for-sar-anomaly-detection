from abc import ABC, abstractmethod
from argparse import Namespace
from typing import Sequence, List, Any
from pathlib import Path
import os

import torch
from torch import nn, Tensor

from lightning import LightningModule
from torchvision.utils import make_grid

from models.utils import MinMaxDenormalize


class BaseModel(LightningModule, ABC):
    
    def __init__(self, opt: Namespace, image_out_dir: str):
        super().__init__()

        self.opt = opt
        self.image_save_dir = Path(f'{image_out_dir}')

        self.denorm = MinMaxDenormalize
    
    def get_name_ext(self, filepath: str, add_epoch: bool = True) -> Sequence[str]:
        filename = os.path.basename(filepath)
        name, ext = os.path.splitext(filename)
        if add_epoch:
            name = f'{name}_epoch_{self.current_epoch}'
        
        return name, ext
    
    def log_image(self, label: str, data: Tensor) -> None:
        # assert logger_id < len(self.loggers), "Invalid logger id"
        grid = make_grid(data)
        self.loggers[0].experiment.add_image(
            label, grid, self.current_epoch
        )
        
    def compute_scm_smv(self, x: Tensor) -> Sequence[Tensor]: 
        mu = x.mean(dim=0)
        x_centered = (x - mu).T
        (p, N) = x.shape
        sigma = (x_centered @ x_centered.conj().T) / (N-1)
        return sigma, mu
    
    def create_anomaly_map(self, pred: Tensor, input: Tensor) -> Tensor:
        c, h, w = pred.shape
        half_kernel = self.opt.recon_anomaly_kernel // 2
        anomaly_map = torch.zeros(h, w)
        for i in range(h):
            for j in range(w):
                up = max(0, i - half_kernel)
                down = min(h, i + half_kernel + 1)
                left = max(0, j - half_kernel)
                right = min(w, j + half_kernel + 1)
                
                pred_patch = pred[:, up:down, left:right]
                pred_patch = pred_patch.reshape(c, -1)
                pred_patch_cov, pred_patch_mean = self.compute_scm_smv(pred_patch)
                
                input_patch = input[:, up:down, left:right]
                input_patch = input_patch.reshape(c, -1)
                input_patch_cov, input_patch_mean = self.compute_scm_smv(input_patch)
                
                anomaly_map[i, j] = torch.linalg.norm(pred_patch_cov - input_patch_cov, ord='fro') ** 2
        
        # anomaly_map = torch.log(anomaly_map + 1e-10)
        
        # min_val = anomaly_map.min()
        # max_val = anomaly_map.max()
        # return (anomaly_map - min_val) / (max_val - min_val)
        return anomaly_map

    
class BaseVAE(nn.Module):
    
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass
