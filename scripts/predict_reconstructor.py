import os, shutil, glob
from argparse import ArgumentParser
from typing import Callable

import torch
from lightning import Trainer

import utils as U
from models import M_M


def reconstructor_predict(opt: ArgumentParser, trainer: Callable) -> None:
    ckpt_path = glob.glob(f'weights_storage/version_{opt.version}/reconstructor/*.ckpt')
    ckpt_path = next((p for p in ckpt_path if f"{opt.data_band}-band" in p), None)
    if ckpt_path:
        # Define data module
        data_module = U.ReconstructionDatasetModule(opt)
        data_module.setup(stage='predict')
        image_out_dir = data_module.pred_dataset.data_dir.replace('despeckled', 'reconstructed')
        if opt.recon_model == 'vae':
            model = M_M.VAEModule.load_from_checkpoint(ckpt_path, opt=opt, image_out_dir=image_out_dir)
        elif opt.recon_model == 'aae':
            model = M_M.AAEModule.load_from_checkpoint(ckpt_path, opt=opt, image_out_dir=image_out_dir)
        # Predict
        trainer.predict(model, datamodule=data_module)
    else:
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = U.ArgumentParsing(parser)
    parser.predict_reconstructor_args(parser.predict_reconstructor_group)
    opt = parser.parser.parse_args()

    if opt.recon_anomaly_kernel % 2 == 0:
        raise ValueError("Anomaly kernel size must be an odd number")
    
    torch.backends.cudnn.enabled = True
    torch.set_float32_matmul_precision('high')
    
    workdir = os.getenv('RECONSTRUCTOR_WORKDIR', '')
    reconstructor = Trainer(
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        benchmark=True,
        logger=False,
        callbacks=U.CustomProgressBar()
    )
    image_out_dir = reconstructor_predict(opt, reconstructor)
