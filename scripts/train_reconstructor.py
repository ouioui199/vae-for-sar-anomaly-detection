import os
from argparse import ArgumentParser
from typing import Callable
from pathlib import Path

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

import utils as U
from models import M_M, M_U
    
    
def train_reconstructor(opt: ArgumentParser, trainer: Callable, log_dir: str) -> None:
    # Define reconstruction module
    if opt.recon_model == 'vae':
        reconstruction_module = M_M.VAEModule(opt, log_dir + '/validation_samples')
    elif opt.recon_model == 'aae':
        reconstruction_module = M_M.AAEModule(opt, log_dir + '/validation_samples')
    # Define data module
    data_module = U.ReconstructionDatasetModule(opt)
    data_module.setup(stage='fit')
    if opt.recon_visualize:
        outpath = str(Path(f'{log_dir}/visualize'))
        M_U.check_path(outpath)
        U.visualize_recon(data_module.train_dataloader(), outpath)

    trainer.fit(reconstruction_module, datamodule=data_module)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = U.ArgumentParsing(parser)
    parser.train_reconstructor_args(parser.train_reconstructor_group)
    opt = parser.parser.parse_args()
    
    torch.backends.cudnn.enabled = True
    torch.set_float32_matmul_precision('high')
    
    workdir = os.getenv('RECONSTRUCTOR_WORKDIR', '')
    
    image_output_dir = Path(workdir)/f'{opt.logdir}/version_{str(opt.version)}/reconstructor'
    os.makedirs(image_output_dir, exist_ok=True)
    trainer = Trainer(
        max_epochs=opt.recon_epochs,
        num_sanity_val_steps=0,
        benchmark=True,
        enable_progress_bar=False,
        callbacks=[
            LearningRateMonitor(logging_interval='epoch'),
            # U.CustomProgressBar(),
            ModelCheckpoint(
                dirpath=Path(workdir)/f'weights_storage/version_{str(opt.version)}/reconstructor',
                filename=f'{{epoch}}_{{step}}_{opt.data_band}-band',
                monitor="val_psnr",
                verbose=True,
                save_on_train_epoch_end=True,
                mode='max'
            ),
            EarlyStopping(
                monitor='val_rec_loss',
                patience=150,
                verbose=True,
                check_on_train_epoch_end=True,
                min_delta=1e-6
            )
        ],
        logger=[
            U.TBLogger(Path(workdir)/'training_logs', name=None, version=f'version_{opt.version}', sub_dir='reconstructor/train'),
            U.TBLogger(Path(workdir)/'training_logs', name=None, version=f'version_{opt.version}', sub_dir='reconstructor/valid'),
        ]
    )
    train_reconstructor(opt, trainer, str(image_output_dir))
