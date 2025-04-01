from typing import Dict
from types import NoneType
from argparse import ArgumentParser
from overrides import override

import torch
import numpy as np
from torch import Tensor

from datasets.base_dataset import ReconstructorFile, DespecklerFile, ReconstructorDataset, DespecklerDataset
from datasets.utils import symetrisation_patch

#TODO a voir si c'est important de préciser phase ou pas dans AAETrain, AAEValid
class ReconstructorTrainFile(ReconstructorFile):
    def __init__(
        self, 
        opt: ArgumentParser,
        filepath: str,
        filepath_rx: str | NoneType,
        phase: str = 'train',
    ) -> None:
        super().__init__(opt, filepath, filepath_rx, phase)

        _, nrows, ncols = self.data.shape
        self.nsamples_per_rows = (ncols - opt.recon_patch_size) // opt.recon_stride + 1
        self.nsamples_per_cols = (nrows - opt.recon_patch_size) // opt.recon_stride + 1
    
    @override
    def __len__(self) -> int:
        return self.nsamples_per_rows * self.nsamples_per_cols
    
    @override
    def __getitem__(self, index: int) -> Dict[str, np.memmap | str | int]:
        row = index // self.nsamples_per_rows
        col = index % self.nsamples_per_rows

        row_start = row * self.opt.recon_stride
        col_start = col * self.opt.recon_stride

        output = {
            'data': self.data[:, row_start : row_start + self.opt.recon_patch_size, col_start : col_start + self.opt.recon_patch_size],
            'filepath': self.filepath,
            'min': self.min_val,
            'max': self.max_val
        }
        
        if self.opt.recon_conditional:
            output['RX_map'] = self.rx_map[:, row_start : row_start + self.opt.recon_patch_size, col_start : col_start + self.opt.recon_patch_size]

        if self.opt.recon_visualize:
            output['row'] = row
            output['col'] = col

        return output
    

class ReconstructorValidFile(ReconstructorFile):

    def __init__(
        self, 
        opt: ArgumentParser, 
        filepath: str,
        filepath_rx: str | NoneType,
        phase: str = 'valid',
    ) -> None:
        super().__init__(opt, filepath, filepath_rx, phase)

    @override
    def __len__(self) -> int:
        return 1
    
    @override
    def __getitem__(self, index: int) -> Dict[str, np.memmap | str]:
        output = {
            'data': self.data,
            'filepath': self.filepath,
            'min': self.min_val,
            'max': self.max_val
        }

        return output
    

class ReconstructorTrainDataset(ReconstructorDataset):

    def __init__(
            self, 
            opt: ArgumentParser,
            phase: str = 'train'
    ) -> None:
        super().__init__(opt, phase)
            
        self.filepath = next((s for s in self.filepath if 'crop' not in s), None)
        self.filepath_rx = next((s for s in self.filepath_rx if 'crop' not in s), None)

        self.dataset = ReconstructorTrainFile(opt, self.filepath, self.filepath_rx, phase)
        self.normalize = self.get_normalization(self.dataset)

    @override
    def __len__(self) -> int:
        return len(self.dataset)

    @override
    def __getitem__(self, index: int) -> Dict[str, Tensor | str | int]:
        image = self.normalize(self.dataset[index]['data'])
        
        output = {
            'image': self.to_tensor(image),
            'filepath': self.dataset[index]['filepath'],
            'min': self.to_tensor(self.dataset.min_val),
            'max': self.to_tensor(self.dataset.max_val)
        }

        if self.opt.recon_conditional:
            output['RX_map'] = self.to_tensor(self.dataset[index]['RX_map'])

        if self.opt.recon_visualize:
            output['row'] = str(self.dataset[index]['row'])
            output['col'] = str(self.dataset[index]['col'])

        return output
    

class ReconstructorValidDataset(ReconstructorDataset):

    def __init__(
        self, 
        opt: ArgumentParser,
        phase: str = 'valid'
    ) -> None:
        super().__init__(opt, phase)
        filepath = next((s for s in self.filepath if ('crop' not in s)), None)
        filepath_rx = next((s for s in self.filepath_rx if ('crop' not in s)), None)
        self.dataset = ReconstructorValidFile(opt, filepath, filepath_rx, phase)
        self.normalize = self.get_normalization(self.dataset)

    @override
    def __len__(self) -> int:
        return len(self.dataset)
    
    @override
    def __getitem__(self, index: int) -> Dict[str, Tensor | str | int]:
        filepath = self.dataset[index]['filepath']
        image = self.dataset[index]['data']
        image = self.normalize(image)
        output = {
            'image': self.to_tensor(image),
            'filepath': filepath,
            'min': self.to_tensor(self.dataset.min_val),
            'max': self.to_tensor(self.dataset.max_val)
        }

        return output
    

class ReconstructorPredictDataset(ReconstructorDataset):
    
    def __init__(
        self, 
        opt: ArgumentParser,
        phase: str = 'predict'
    ) -> None:
        super().__init__(opt, phase)
        
        filepath = self.filepath
        if self.opt.recon_sample_prediction:
            filepath = [s for s in self.filepath if 'crop' in s]
        
        self.dataset = [ReconstructorValidFile(opt, fp, None, phase) for fp in filepath]
    
    @override
    def __len__(self) -> int:
        return len(self.dataset)

    @override
    def __getitem__(self, index: int) -> Dict[str, Tensor | str | int]:
        dataset = self.dataset[index]
        normalize = self.get_normalization(dataset)
        image = normalize(dataset[0]['data'])
        
        output = {
            'image': self.to_tensor(image),
            'filepath': dataset[0]['filepath'],
            'min': self.to_tensor(dataset.min_val),
            'max': self.to_tensor(dataset.max_val)
        }

        return output


class DespecklerTrainFile(DespecklerFile):

    def __init__(
        self, 
        opt: ArgumentParser,
        filepath: str, 
        phase: str = 'train'
    ) -> None:
        """Return one of its patches from the data memory map for training."""
        super().__init__(opt, filepath, phase)

        nrows, ncols = self.data.shape
        self.nsamples_per_rows = (ncols - opt.despeckler_patch_size) // opt.despeckler_stride + 1
        self.nsamples_per_cols = (nrows - opt.despeckler_patch_size) // opt.despeckler_stride + 1
    
    @override
    def __len__(self) -> int:
        """Returns the number of patches that can be extracted from the NPY file."""
        return self.nsamples_per_rows * self.nsamples_per_cols
    
    @override
    def __getitem__(self, index: int) -> Dict[str, np.memmap | str | int]:
        """Returns the index-th patch from the NPY file.

        Args:
            item (int): index of the patch to be extracted.

        Returns:
            Dict[str, np.memmap | str | int]: Numpy memory map data and metadata
        """
        row = index // self.nsamples_per_rows
        col = index % self.nsamples_per_rows

        row_start = row * self.opt.despeckler_stride
        col_start = col * self.opt.despeckler_stride

        output = {
            'data': self.data[row_start : row_start + self.opt.despeckler_patch_size, col_start : col_start + self.opt.despeckler_patch_size],
            'filepath': self.filepath
        }

        if self.opt.despeckler_visualize:
            output['row'] = row
            output['col'] = col

        return output


class DespecklerValidFile(DespecklerFile):

    def __init__(
        self, 
        opt: ArgumentParser, 
        filepath: str, 
        phase: str = 'valid'
    ) -> None:
        """Return the whole validation part of the image"""
        super().__init__(opt, filepath, phase)

    @override
    def __len__(self) -> int:
        """Returns 1 as NPYFile process individual image."""
        return 1
    
    @override
    def __getitem__(self, index: int) -> Dict[str, np.memmap | str | int]:
        """Returns validation data from the NPY file.

        Args:
            item (int): index of the patch to be extracted.

        Returns:
            Dict[str, np.memmap | str | int]: Numpy memory map data and metadata
        """
        output = {
            'data': self.data,
            'filepath': self.filepath
        }

        return output


class DespecklerTrainDataset(DespecklerDataset):

    def __init__(
            self, 
            opt: ArgumentParser,
            phase: str = 'train'
    ) -> None:
        """Form a training dataset."""
        super().__init__(opt, phase)
        
        if len(self.filepath) != 1:
            raise RuntimeError(f"Only one image with 4 polarizations is allowed in the training dataset. Found {len(self.filepath)} files.")
        self.dataset = DespecklerTrainFile(opt, self.filepath[0], phase)
        self.normalize = self.normalize(self.dataset.data_min_max['min'], self.dataset.data_min_max['max'])

    @override
    def __len__(self) -> int:
        """Returns the total number of patches of the dataset."""
        return len(self.dataset)

    @override
    def __getitem__(self, index: int) -> Dict[str, Tensor | str | int]:
        """Returns the index-th patch of the associated NPY file.

        Args:
            index (int): index of the patch to be extracted

        Returns:
            Dict[str, Tensor | str | int]: Tensor and metadata
        """
        real, imag = self.normalize(self.dataset[index]['data'].real), self.normalize(self.dataset[index]['data'].imag)
        real, imag = symetrisation_patch(real, imag)
        output = {
            'real': self.to_tensor(real),
            'imag': self.to_tensor(imag),
            'filepath': self.dataset[index]['filepath'],
            'min': self.dataset.data_min_max['min'].astype(np.float32),
            'max': self.dataset.data_min_max['max'].astype(np.float32)
        }

        if self.opt.despeckler_visualize:
            output['row'] = str(self.dataset[index]['row'])
            output['col'] = str(self.dataset[index]['col'])

        return output
    

class DespecklerValidDataset(DespecklerDataset):

    def __init__(
        self, 
        opt: ArgumentParser, 
        phase: str = 'valid'
    ) -> None:
        """Form a validation dataset."""
        super().__init__(opt, phase)

        if len(self.filepath) != 1:
            raise RuntimeError(f"Only one image with 4 polarizations is allowed in the validation dataset. Found {len(self.filepath)} files.")
        self.dataset = DespecklerValidFile(opt, self.filepath[0], phase)
        self.dataset_min_max = self.dataset.data_min_max
        
        self.normalize = self.normalize(self.dataset_min_max['min'], self.dataset_min_max['max'])

    @override
    def __len__(self) -> int:
        return len(self.dataset)
    
    @override
    def __getitem__(self, index: int) -> Dict[str, Tensor | str | int]:
        real, imag = self.normalize(self.dataset[index]['data'].real), self.normalize(self.dataset[index]['data'].imag)
        output = {
            'real': torch.from_numpy(real.astype(np.float32)),
            'imag': torch.from_numpy(imag.astype(np.float32)),
            'filepath': self.dataset[index]['filepath'],
            'min': self.dataset.data_min_max['min'].astype(np.float32),
            'max': self.dataset.data_min_max['max'].astype(np.float32)
        }

        return output


class DespecklerPredictDataset(DespecklerDataset):
    
    def __init__(
        self, 
        opt: ArgumentParser,
        phase: str = 'predict'
    ) -> None:
        super().__init__(opt, phase)
        
        if len(self.filepath) != 1:
            self.dataset = [DespecklerValidFile(opt, f, phase) for f in self.filepath]
        else:
            self.dataset = DespecklerValidFile(opt, self.filepath[0], phase)
        
        
    @override
    def __len__(self) -> int:
        return len(self.dataset)
    
    @override
    def __getitem__(self, index: int) -> Dict[str, Tensor | str | int]:
        data = self.dataset[index]
        image_phase = np.angle(data[0]['data'])
        normalize = self.normalize(data.data_min_max['min'], data.data_min_max['max'])
        real, imag = normalize(data[0]['data'].real), normalize(data[0]['data'].imag)
        output = {
            'real': torch.from_numpy(real.astype(np.float32)),
            'imag': torch.from_numpy(imag.astype(np.float32)),
            'filepath': data[0]['filepath'],
            'min': data.data_min_max['min'].astype(np.float32),
            'max': data.data_min_max['max'].astype(np.float32),
            'image_phase': torch.from_numpy(image_phase)
        }

        return output
