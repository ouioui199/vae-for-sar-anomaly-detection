import os, glob
from argparse import ArgumentParser, Namespace
from typing import List, Sequence
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
from lightning import LightningDataModule
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm
from lightning.pytorch.utilities import rank_zero_only
from torch.utils.data import DataLoader
from torchcvnn.transforms.functional import equalize

import datasets.dataset as D
from models.utils import MinMaxDenormalize


class TBLogger(TensorBoardLogger):
    @rank_zero_only
    def log_metrics(self, metrics, step):
        metrics.pop('epoch', None)
        metrics = {k: v for k, v in metrics.items() if ('step' not in k) and ('val' not in k)}
        return super().log_metrics(metrics, step)
    
    
class CustomProgressBar(TQDMProgressBar):
    
    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items
    
    def init_train_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for training."""
        bar = super().init_train_tqdm()
        bar.ascii = ' >'
        return bar
    
    def init_validation_tqdm(self) -> Tqdm:
        bar = super().init_validation_tqdm()
        bar.ascii = ' >'
        return bar
    
    def init_predict_tqdm(self) -> Tqdm:
        bar = super().init_validation_tqdm()
        bar.ascii = ' >'
        return bar


class BaseDataModule(LightningDataModule):
    def __init__(self, opt: Namespace) -> None:
        super().__init__()
        self.opt = opt
        self.pin_memory = False

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size,
            num_workers=self.opt.workers,
            pin_memory=True,
            shuffle=True,
            persistent_workers=True,
            prefetch_factor=20
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_dataset,
            batch_size=1,
            num_workers=self.opt.workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            persistent_workers=True
        )
        
    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.pred_dataset, 
            batch_size=1,
            num_workers=self.opt.workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            persistent_workers=True
        )


class DespecklerDatasetModule(BaseDataModule):
    def __init__(self, opt: Namespace) -> None:
        super().__init__(opt)

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            self.batch_size = self.opt.despeckler_batch_size
            self.train_dataset = D.DespecklerTrainDataset(self.opt)
            self.valid_dataset = D.DespecklerValidDataset(self.opt)
        
        if stage == 'predict':
            self.pred_dataset = D.DespecklerPredictDataset(self.opt)


class ReconstructionDatasetModule(BaseDataModule):
    def __init__(self, opt: Namespace) -> None:
        super().__init__(opt)

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            self.batch_size = self.opt.recon_batch_size
            self.train_dataset = D.ReconstructorTrainDataset(self.opt)
            self.valid_dataset = D.ReconstructorValidDataset(self.opt)
        
        if stage == 'predict':
            self.pred_dataset = D.ReconstructorPredictDataset(self.opt)
            if self.opt.recon_sample_prediction:
                self.pin_memory = True


class ArgumentParsing:
    def __init__(self, parser: ArgumentParser) -> None:
        self.parser = parser
        self.common_args()
        self.train_despeckler_group = self.parser.add_argument_group()
        self.predict_despeckler_group = self.parser.add_argument_group()
        self.compute_rx_group = self.parser.add_argument_group()
        self.train_reconstructor_group = self.parser.add_argument_group()
        self.predict_reconstructor_group = self.parser.add_argument_group()
        
    def common_args(self) -> None:
        self.parser.add_argument('--version', type=str, required=True)
        self.parser.add_argument('--workers', type=int, default=4)
        self.parser.add_argument('--datadir', type=str, required=True)
        self.parser.add_argument('--logdir', type=str, default='training_logs')
        self.parser.add_argument('--train_valid_ratio', type=List, default=[0.8, 0.2])
        self.parser.add_argument('--data_band', type=str, default=None)
        self.parser.add_argument('--rx_box_car_size', type=int, default=39)
        self.parser.add_argument('--rx_exclusion_window_size', type=int, default=31)
        
    def train_despeckler_args(self, group: ArgumentParser) -> None:
        group.add_argument('--despeckler_visualize', action='store_true')
        group.add_argument('--despeckler_lr', type=float, default=1e-3)
        group.add_argument('--despeckler_epochs', type=int, default=50)
        group.add_argument('--despeckler_batch_size', type=int, default=32)
        group.add_argument('--despeckler_patch_size', type=int, default=256)
        group.add_argument('--despeckler_stride', type=int, default=300)
        group.add_argument('--despeckler_pol_channels', type=str, default='Hh', choices=['Hh', 'Hv', 'Vh', 'Vv'])

    def train_reconstructor_args(self, group: ArgumentParser) -> None:
        group.add_argument('--recon_visualize', action='store_true')
        group.add_argument('--recon_model', required=True, type=str, choices=['aae', 'vae'], default='vae')
        group.add_argument('--recon_train_slc', action='store_true')
        group.add_argument('--recon_patch_size', type=int, default=32)
        group.add_argument('--recon_stride', type=int, default=16)
        group.add_argument('--recon_in_channels', type=int, default=4)
        group.add_argument('--recon_keep_phase', action='store_false')
        group.add_argument('--recon_batch_size', type=int, default=128)
        group.add_argument('--recon_epochs', type=int, default=100)
        group.add_argument('--recon_latent_size', type=int, default=128)
        group.add_argument('--recon_lr_ae', type=float, default=1e-3)
        group.add_argument('--recon_lr_gen', type=float, default=1e-3)
        group.add_argument('--recon_lr_disc', type=float, default=1e-3)
        group.add_argument('--recon_beta_start', type=float, default=0.)
        group.add_argument('--recon_beta_end', type=float, default=1.)
        group.add_argument('--recon_beta_proportion', type=float, default=.8)
        group.add_argument('--recon_beta_warmup_epochs', type=int, default=5)
        group.add_argument('--recon_conditional', action='store_true')
        
        group.add_argument('--recon_sample_prediction', action='store_true')
        group.add_argument('--recon_anomaly_kernel', type=int, default=11)
        
    def predict_reconstructor_args(self, group: ArgumentParser) -> None:
        group.add_argument('--recon_sample_prediction', action='store_true')
        group.add_argument('--recon_in_channels', type=int, default=4)
        group.add_argument('--recon_latent_size', type=int, default=128)
        group.add_argument('--recon_anomaly_kernel', type=int, default=11)
        group.add_argument('--recon_patch_size', type=int, default=32)
        group.add_argument('--recon_stride', type=int, default=16)


def visualize_recon(dataloader: DataLoader, outpath: str) -> None:
    denorm = MinMaxDenormalize
    for data in tqdm(dataloader):
        for i in range(len(data['filepath'])):
            filepath = data['filepath'][i]
            (pos_row, pos_col) = (data['row'][i], data['col'][i]) if 'row' in data else (0, 0)
            data_min, data_max = data['min'][i].numpy(), data['max'][i].numpy()
        
            image_denorm = denorm(data_min, data_max)(data['image'][i].numpy())
            image = equalize(image_denorm.transpose(1, 2, 0).squeeze(), plower=0, pupper=100)
            image = Image.fromarray(image)

            filename = os.path.basename(filepath)
            name, ext = os.path.splitext(filename)
            filename = f'{name}_{pos_row}_{pos_col}'
            image.save(Path(f'{outpath}/{filename}.png'))

            np.save(Path(f'{outpath}/{filename}{ext}'), image_denorm)
            

def visualize_despeckler(dataloader: DataLoader, outpath: str) -> None:
    denorm = MinMaxDenormalize

    for data in tqdm(dataloader):
        for i in range(len(data['filepath'])):
            filepath = data['filepath'][i]
            (pos_row, pos_col) = (data['row'][i], data['col'][i]) if 'row' in data else (0, 0)
            data_min, data_max = data['min'][i].numpy(), data['max'][i].numpy()

            image_denorm = denorm(data_min, data_max)(data['real'][i].numpy()) + 1j * denorm(data_min, data_max)(data['imag'][i].numpy())

            image = equalize(image_denorm.transpose(1, 2, 0).squeeze())
            image = Image.fromarray(image)

            filename = os.path.basename(filepath)
            name, ext = os.path.splitext(filename)
            filename = f'{name}_{pos_row}_{pos_col}'
            image.save(Path(f'{outpath}/{filename}.png'))

            np.save(Path(f'{outpath}/{filename}{ext}'), image_denorm)


def combine_polar_channels(paths: Sequence[str]) -> str:
    # Find paths containing specific polarizations
    path_hh = next((p for p in paths if 'Hh' in p), None)
    path_hv = next((p for p in paths if 'Hv' in p), None)
    path_vh = next((p for p in paths if 'Vh' in p), None)
    path_vv = next((p for p in paths if 'Vv' in p), None)
    path_combined = next((p for p in paths if 'Combined' in p), None)
    if path_combined is not None:
        print('Combined polarization image already exists. Skipping...')
        return 
    
    data = []
    pol_channels = []
    if path_hh:
        data_hh = np.load(path_hh, mmap_mode='r')
        data.append(data_hh)
        pol_channels.append('Hh')
        del data_hh
    else:
        print('Hh polarization not found')
        
    if path_hv:
        data_hv = np.load(path_hv, mmap_mode='r')
        data.append(data_hv)
        pol_channels.append('Hv')
        del data_hv
    else:
        print('Vh polarization not found')
        
    if path_vh:
        data_vh = np.load(path_vh, mmap_mode='r')
        data.append(data_vh)
        pol_channels.append('Vh')
        del data_vh
    else:
        print('Vh polarization not found')
        
    if path_vv:
        data_vv = np.load(path_vv, mmap_mode='r')
        data.append(data_vv)
        pol_channels.append('Vv')
        del data_vv
    else:
        print('Vv polarization not found')
    
    if len(pol_channels) != 0:
        print(f'Combining {pol_channels} polarizations')
        data_combined = np.stack(data, axis=0)
        print(f'Saving combined polarization image to {path_hh.replace("Hh", "Combined")}')
        path_combined = path_hh.replace('Hh', 'Combined')
        np.save(path_combined, data_combined)
    else:
        raise FileNotFoundError('No polarization channels found in the data directory')
    
    return path_combined
