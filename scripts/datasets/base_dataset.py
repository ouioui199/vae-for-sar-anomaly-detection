from typing import Any, Dict
from pathlib import Path
from argparse import ArgumentParser
from abc import ABC, abstractmethod
from types import NoneType
import os, glob

import numpy as np
from torch.utils.data import Dataset
from torchcvnn.transforms import ToTensor

import datasets.utils as D_U
import datasets.transforms as T


class ReconstructorFile(ABC):

    def __init__(
        self, 
        opt: ArgumentParser,
        filepath: str,
        filepath_rx: str | NoneType,
        phase: str
    ) -> None:
        self.opt = opt
        self.filepath = filepath
        self.phase = phase
        # Load image
        self.data = np.load(filepath, mmap_mode='r')
        self.data = np.abs(self.data)
        if hasattr(opt, 'recon_conditional') and opt.recon_conditional and filepath_rx is not None:
            self.rx_map = np.load(filepath_rx, mmap_mode='r')
            self.rx_map = self.get_train_valid('data_rx')
            self.data = self.data[:, opt.rx_box_car_size // 2 : -opt.rx_box_car_size // 2, opt.rx_box_car_size // 2 : -opt.rx_box_car_size // 2]
            assert self.data.shape == self.rx_map.shape, f"Image and RX map shape mismatch: {self.data.shape} != {self.rx_map.shape}"
        
        self.min_val, self.max_val = self.get_min_max()
        self.min_val, self.max_val = self.min_val.reshape(-1, 1, 1), self.max_val.reshape(-1, 1, 1)
        # Ensure CHW format
        self.data = D_U.ensure_chw_format(self.data)
        self.data = self.get_train_valid('data')
                    
    def get_train_valid(self, data_name: str) -> Dict[str, np.ndarray]:
        _, h, w = self.data.shape
        train_valid_threshold = int(max(h, w) * self.opt.train_valid_ratio[0])
        data = getattr(self, data_name)
        if h > w:
            train_valid = {
                'train': data[:, :train_valid_threshold, :],
                'valid': data[:, train_valid_threshold:, :],
                'predict': data
            }
        else:
            train_valid = {
                'train': data[:, :, :train_valid_threshold],
                'valid': data[:, :, train_valid_threshold:],
                'predict': data
            }
        
        for (k, v) in train_valid.items():
            if k == self.phase:
                return v
                
    def get_min_max(self) -> Dict[str, float]:
        data = np.log(self.data + np.spacing(1))
        min_val, max_val = data.min(axis=(-2, -1)), data.max(axis=(-2, -1))
        if hasattr(self.opt, 'recon_train_slc') and self.opt.recon_train_slc:
            min_val, max_val = np.percentile(data, (5, 95))
        return min_val, max_val
    
    @abstractmethod
    def __len__(self) -> int:
        return 0
    
    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        pass
    

class DespecklerFile(ABC):

    def __init__(
        self, 
        opt: ArgumentParser,
        filepath: str, 
        phase: str
    ) -> None:
        self.opt = opt
        self.filepath = filepath
        
        self.data = np.load(filepath, mmap_mode='r').squeeze()
        h, w = self.data.shape
        self.data_min_max = self.get_min_max()

        train_valid_threshold = int(max(h, w) * opt.train_valid_ratio[0])
        if h > w:
            train_valid = {
                'train': self.data[:train_valid_threshold, :],
                'valid': self.data[train_valid_threshold:, :]
            }
        else:
            train_valid = {
                'train': self.data[:, :train_valid_threshold],
                'valid': self.data[:, train_valid_threshold:]
            }
        for (k,v) in train_valid.items():
            if k == phase:
                self.data = v

    def get_min_max(self) -> Dict[str, float]:
        data = np.log(self.data.__abs__() + np.spacing(1))
        min_val, max_val = np.percentile(data, (5, 95))
        return {
            'min': min_val,
            'max': max_val,
        }
    
    @abstractmethod
    def __len__(self) -> int:
        return 0
    
    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        pass


class BaseDataset(ABC, Dataset):
    def __init__(
            self, 
            opt: ArgumentParser,
            phase: str
    ) -> None:
        super().__init__()

        assert phase in ['train', 'valid', 'predict'], "Dataset only accept 'train', 'valid', 'predict' as phase"
        assert sum(opt.train_valid_ratio) == 1., "Train and valid proportion does not cover all image"
        
        self.opt = opt
        self.data_dir = os.path.join(opt.datadir, opt.data_band + '_band') if opt.data_band else opt.datadir
        self.data_dir = os.path.join(self.data_dir, 'train' if phase in ['train', 'valid'] else 'predict')
        self.normalize = T.MinMaxNormalize
        self.to_tensor = ToTensor('float32')
    
    @staticmethod
    def get_normalization(dataset: Dataset) -> None:
        return T.MinMaxNormalize(dataset.min_val, dataset.max_val)

    @abstractmethod
    def __len__(self) -> int:
        return 0

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        pass
    

class DespecklerDataset(BaseDataset):
    def __init__(
        self, 
        opt: ArgumentParser,
        phase: str
    ) -> None:
        super().__init__(opt, phase)
        self.data_dir = os.path.join(self.data_dir, 'slc')
        if D_U.is_directory_empty(self.data_dir):
            raise FileNotFoundError(f"{self.data_dir} is empty or doesn't exist.")
        # Get all .npy files in data directory
        filepath = glob.glob(f'{self.data_dir}/*.npy')
        # Filter files based on polarization channel
        if not filepath:
            raise FileNotFoundError(f"No files found with {opt.despeckler_pol_channels} polarization in {self.data_dir}")
        self.filepath = [s for s in filepath if opt.despeckler_pol_channels in s]


class ReconstructorDataset(BaseDataset):
    def __init__(
        self, 
        opt: ArgumentParser,
        phase: str
    ) -> None:
        super().__init__(opt, phase)
        self.data_dir = os.path.join(self.data_dir, 'despeckled')
        datadir_rx = self.data_dir.replace('despeckled', 'slc')
        if hasattr(opt, 'recon_train_slc') and opt.recon_train_slc:
            self.data_dir = datadir_rx
        #TODO check datadir_rx exists or empty
        if D_U.is_directory_empty(self.data_dir):
            raise FileNotFoundError(f"data_despeckled directory is empty or doesn't exist.")
        
        filepath = glob.glob(f'{self.data_dir}/*.npy')
        filepath_rx = glob.glob(f'{datadir_rx}/*.npy')
        
        self.filepath = [s for s in filepath if (opt.data_band in s) and ('Combine' in s)]
        self.filepath_rx = [s for s in filepath_rx if (opt.data_band in s) and ('RX-map' in s)]
