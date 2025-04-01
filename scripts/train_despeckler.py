import os
from argparse import ArgumentParser
from typing import Callable
from pathlib import Path

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

import utils as U
from models import M_M, M_U


def train_despeckler(opt: ArgumentParser, log_dir: str, trainer: Callable) -> None:
    # Define despeckler module
    despeckler_module = M_M.MERLINModule(opt, log_dir + f'/validation_samples/{opt.despeckler_pol_channels}')
    # Define data module
    data_module = U.DespecklerDatasetModule(opt)
    data_module.setup(stage='fit')
    if opt.despeckler_visualize:
        outpath = str(Path(f'{log_dir}/visualize'))
        M_U.check_path(outpath)
        U.visualize_despeckler(data_module.train_dataloader(), outpath)
    # Fit model
    trainer.fit(despeckler_module, datamodule=data_module)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = U.ArgumentParsing(parser)
    parser.train_despeckler_args(parser.train_despeckler_group)
    opt = parser.parser.parse_args()
    
    torch.backends.cudnn.enabled = True
    torch.set_float32_matmul_precision('high')
    
    workdir = os.getenv('RECONSTRUCTOR_WORKDIR', '')
    image_output_dir = Path(workdir) / f"{opt.logdir}/version_{str(opt.version)}/despeckler"
    os.makedirs(image_output_dir, exist_ok=True)
    trainer = Trainer(
        max_epochs=opt.despeckler_epochs,
        num_sanity_val_steps=0,
        gradient_clip_val=1.,
        benchmark=True,
        enable_progress_bar=False,
        check_val_every_n_epoch=10,
        callbacks=[
            # U.CustomProgressBar(),
            LearningRateMonitor(logging_interval='epoch'),
            ModelCheckpoint(
                dirpath=Path(workdir)/f'weights_storage/version_{str(opt.version)}/despeckler',
                filename=f'{{epoch}}_{{step}}_{opt.despeckler_pol_channels}_{opt.data_band}-band',
                monitor="loss",
                verbose=True,
                save_on_train_epoch_end=True
            ),
            EarlyStopping(
                monitor='loss',
                patience=5,
                verbose=True,
                check_on_train_epoch_end=True,
                min_delta=1e-6
            )
        ],
        logger=U.TBLogger(Path(workdir)/'training_logs', name=None, version=opt.version, sub_dir='despeckler')
    )
    train_despeckler(opt, str(image_output_dir), trainer)
