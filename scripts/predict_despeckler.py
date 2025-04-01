import os, shutil, glob
from argparse import ArgumentParser
from typing import Callable

import torch
from lightning import Trainer

import utils as U
from models import M_M


def despeckler_predict(opt: ArgumentParser, trainer: Callable) -> None:
    ckpt_path = glob.glob(f'weights_storage/version_{opt.version}/despeckler/*.ckpt')
    ckpt_path = next((p for p in ckpt_path if (opt.despeckler_pol_channels in p) and (f"{opt.data_band}-band" in p)), None)
    if ckpt_path:
        # Define data module
        data_module = U.DespecklerDatasetModule(opt)
        data_module.setup(stage='predict')
        image_out_dir = data_module.pred_dataset.data_dir.replace('slc', 'despeckled')
        model = M_M.MERLINModule.load_from_checkpoint(ckpt_path, opt=opt, image_out_dir=image_out_dir)
        # Predict
        trainer.predict(model, datamodule=data_module)
    else:
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    return image_out_dir


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = U.ArgumentParsing(parser)
    opt = parser.parser.parse_args()
    
    torch.backends.cudnn.enabled = True
    torch.set_float32_matmul_precision('high')
    
    workdir = os.getenv('RECONSTRUCTOR_WORKDIR', '')
    despeckler = Trainer(
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        benchmark=True,
        logger=False,
        callbacks=U.CustomProgressBar()
    )
    for pol in ['Hh', 'Hv', 'Vh', 'Vv']:
        opt.despeckler_pol_channels = pol
        image_out_dir = despeckler_predict(opt, despeckler)
        
    # Combine polar channels
    paths = glob.glob(f'{image_out_dir}/*.npy')
    paths = [p for p in paths if 'crop' not in p]
    path_combined = U.combine_polar_channels(paths, opt.data_band)
    # Copy combined image to training directory
    shutil.copyfile(path_combined, path_combined.replace('predict', 'train'))
