from argparse import Namespace
from typing import Sequence, List, Dict
from pathlib import Path

import torch
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from torch import nn, Tensor
from torch.nn import MSELoss, L1Loss, HuberLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR, CyclicLR, ExponentialLR, ReduceLROnPlateau
from torchcvnn.transforms.functional import equalize
from PIL import Image
from torchmetrics import MetricCollection
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from models.networks import ConvEncoder, ConvDecoder, Discriminator, UNet_v2, VanillaVAE2
from models.loss import l1_loss, disc_loss, gen_loss, kullback_leibler_divergence_loss, SSIMLoss
from models.utils import torch_symetrisation_patch, check_path
from models.base_model import BaseModel


class AAEModule(BaseModel):

    def __init__(self, opt: Namespace, image_out_dir: str) -> None:
        super().__init__(opt, image_out_dir)
        self.configure_model()

        self.reconstruction_loss = L1Loss()
        self.discriminator_loss = disc_loss()
        self.generator_loss = gen_loss()

        self.automatic_optimization = False

        self.train_step_outputs = {}
        self.valid_step_outputs = {}

        self.metrics = MetricCollection({
            'psnr': PeakSignalNoiseRatio(),
            'ssim': StructuralSimilarityIndexMeasure()
        })

    def on_fit_start(self):
        check_path(self.image_save_dir)
        
    def configure_model(self) -> None:
        self.encoder = ConvEncoder(
            im_ch=self.opt.recon_in_channels,
            nz=self.opt.recon_latent_size,
            patch_size=self.opt.recon_patch_size
        )
        self.decoder = ConvDecoder(
            im_ch=self.opt.recon_in_channels,
            nz=self.opt.recon_latent_size,
            patch_size=self.opt.recon_patch_size
        )
        self.discriminator = Discriminator(nz=self.opt.recon_latent_size)

    def configure_optimizers(self) -> Sequence[torch.optim.Optimizer]:
        encoder_optimizer = Adam(self.encoder.parameters(), lr=self.opt.recon_lr_ae)
        decoder_optimizer = Adam(self.decoder.parameters(), lr=self.opt.recon_lr_ae)
        generator_optimizer = Adam(self.encoder.parameters(), lr=self.opt.recon_lr_gen)
        discriminator_optimizer = Adam(self.discriminator.parameters(), lr=self.opt.recon_lr_disc)
        # Define schedulers
        encoder_scheduler = CyclicLR(encoder_optimizer, base_lr=0.001, max_lr=0.01,step_size_up=6948,cycle_momentum=False)
        decoder_scheduler = CyclicLR(decoder_optimizer, base_lr=0.001, max_lr=0.01,step_size_up=6948,cycle_momentum=False)
        generator_scheduler = CyclicLR(generator_optimizer, base_lr=0.001, max_lr=0.01,step_size_up=6948,cycle_momentum=False)
        discriminator_scheduler = CyclicLR(discriminator_optimizer, base_lr=0.001, max_lr=0.01,step_size_up=6948,cycle_momentum=False)
        return (
            {"optimizer": encoder_optimizer, "lr_scheduler": encoder_scheduler},
            {"optimizer": decoder_optimizer, "lr_scheduler": decoder_scheduler},
            {"optimizer": generator_optimizer, "lr_scheduler": generator_scheduler},
            {"optimizer": discriminator_optimizer, "lr_scheduler": discriminator_scheduler},
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    @staticmethod
    def set_requires_grad(nets: List | nn.Module, requires_grad: bool = False) -> None:
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    
    def training_step(self, batch: Dict[str, str | torch.Tensor], batch_idx: int) -> None:
        image = batch['image']
        encoder_optimizer, decoder_optimizer, generator_optimizer, discriminator_optimizer = self.optimizers()
        encoder_scheduler, decoder_scheduler, generator_scheduler, discriminator_scheduler = self.lr_schedulers()

        # Reconstruction
        self.set_requires_grad(self.discriminator, requires_grad=False)
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        latent = self(image)
        reconstruction = self.decoder(latent)
        reconstruction_loss = 2 * (batch['max'] - batch['min']).mean() * self.reconstruction_loss(reconstruction, image)
        metrics = self.metrics(reconstruction.detach(), image)
        
        self.manual_backward(reconstruction_loss)

        encoder_optimizer.step()
        decoder_optimizer.step()
        
        if batch_idx % 1000 == 0:
            self.log_image("Input", (image * 255).to(torch.uint8))
            self.log_image("Reconstruction", (reconstruction * 255).to(torch.uint8))
            self.log_image("Difference", ((reconstruction - image) * 255).to(torch.uint8))

        # Discriminator
        self.set_requires_grad(self.discriminator, requires_grad=True)
        discriminator_optimizer.zero_grad()

        latent = self(image)
        normal_distribution = torch.randn_like(latent)

        real = self.discriminator(normal_distribution)
        fake = self.discriminator(latent.detach())
        discriminator_loss = self.discriminator_loss(real, fake)
        self.manual_backward(discriminator_loss)

        discriminator_optimizer.step()

        # Generator
        self.set_requires_grad(self.discriminator, requires_grad=False)
        generator_optimizer.zero_grad()
        latent = self(image)
        fake = self.discriminator(latent)
        generator_loss = self.generator_loss(fake)
        self.manual_backward(generator_loss)
        
        generator_optimizer.step()

        encoder_scheduler.step()
        decoder_scheduler.step()
        generator_scheduler.step()
        discriminator_scheduler.step()
        
        if not self.train_step_outputs:
            self.train_step_outputs = {
                "step_rec_loss": [reconstruction_loss], 
                "step_disc_loss": [discriminator_loss],
                "step_gen_loss": [generator_loss],
                "step_metrics_psnr": [metrics['psnr']],
                "step_metrics_ssim": [metrics['ssim']]
            }
        else:
            self.train_step_outputs["step_rec_loss"].append(reconstruction_loss)
            self.train_step_outputs["step_disc_loss"].append(discriminator_loss)
            self.train_step_outputs["step_gen_loss"].append(generator_loss)
            self.train_step_outputs["step_metrics_psnr"].append(metrics['psnr'])
            self.train_step_outputs["step_metrics_ssim"].append(metrics['ssim'])
        
    def on_train_epoch_end(self) -> None:
        tb_logger = self.loggers[0].experiment
        _log_dict = {
            key.replace('step_', ''): torch.tensor(value).mean()
            for (key, value) in self.train_step_outputs.items()
        }

        _log_dict_loss = {f'Loss/{key}': value for (key, value) in _log_dict.items() if 'loss' in key}
        for k,v in _log_dict_loss.items():
            tb_logger.add_scalar(k, v, self.current_epoch)

        _log_dict_metrics = {f'Metrics/{key}'.replace('metrics_', ''): value for (key, value) in _log_dict.items() if 'metrics' in key}
        for k,v in _log_dict_metrics.items():
            tb_logger.add_scalar(k, v, self.current_epoch)
        self.train_step_outputs.clear()

    def _val_step(self, batch: Dict[str, Tensor | str], loop_idx: int) -> Tensor:
        image = batch['image'][loop_idx]
        _, h, w = image.shape

        pred = torch.zeros_like(image)
        patch_overlap_count = torch.zeros_like(pred)

        if h == self.opt.recon_patch_size:
            x_range = list(np.array([0]))
        else:
            x_range = list(range(0, h - self.opt.recon_patch_size, self.opt.recon_stride))
            if (x_range[-1] + self.opt.recon_patch_size) < h :
                x_range.extend(range(h - self.opt.recon_patch_size, h - self.opt.recon_patch_size + 1))
        
        if w == self.opt.recon_patch_size:
            y_range = list(np.array([0]))
        else:
            y_range = list(range(0, w - self.opt.recon_patch_size, self.opt.recon_stride))
            if (y_range[-1] + self.opt.recon_patch_size) < w:
                y_range.extend(range(w - self.opt.recon_patch_size, w - self.opt.recon_patch_size + 1))

        for x in x_range:
            for y in y_range:
                patch = image[:, x : x + self.opt.recon_patch_size, y : y + self.opt.recon_patch_size]
                patch = self.decoder(self(patch.unsqueeze(0))).squeeze(0)
                pred[:, x : x + self.opt.recon_patch_size, y : y + self.opt.recon_patch_size] += patch
                
                patch_overlap_count[:, x : x + self.opt.recon_patch_size, y : y + self.opt.recon_patch_size] += torch.ones_like(patch)
        
        pred = torch.div(pred, patch_overlap_count)
        del patch_overlap_count
        torch.cuda.empty_cache()
        
        return pred, image
    
    def validation_step(self, batch: Dict[str, Tensor | str], batch_idx: int) -> None:
        pred, image = self._val_step(batch, 0)
        reconstruction_loss = 2 * (batch['max'] - batch['min']).mean() * self.reconstruction_loss(pred, image)

        pred = pred.cpu()
        image = image.cpu()
        val_metrics = MetricCollection({
            'psnr': PeakSignalNoiseRatio(),
            'ssim': StructuralSimilarityIndexMeasure()
        })
        metrics = val_metrics(pred.unsqueeze(0), image.unsqueeze(0))

        if not self.valid_step_outputs:
            self.valid_step_outputs = {
                "step_rec_loss": [reconstruction_loss],
                "step_val_metrics_psnr": [metrics['psnr']],
                "step_val_metrics_ssim": [metrics['ssim']]
            }
        else:
            self.valid_step_outputs["step_rec_loss"].append(reconstruction_loss)
            self.valid_step_outputs["step_metrics_psnr"].append(metrics['psnr'])
            self.valid_step_outputs["step_metrics_ssim"].append(metrics['ssim'])

        if self.global_step % (self.trainer.num_training_batches * 20) == 0:
            min, max = batch['min'][0].cpu(), batch['max'][0].cpu()
            denorm = self.denorm(min, max)
            pred = denorm(pred)
            image = denorm(image)

            pred = pred.numpy()
            image = image.numpy()

            name, ext = self.get_name_ext(batch['filepath'][0])
            np.save(Path(f'{self.image_save_dir}/out_{name}.npy'), pred)
            out = Image.fromarray(equalize(pred.transpose(1, 2, 0).squeeze(), plower=0, pupper=100))
            out.save(Path(f'{self.image_save_dir}/out_{name}.png'))
            
            diff = Image.fromarray(equalize((image - pred).transpose(1, 2, 0).squeeze(), plower=0, pupper=100))
            diff.save(Path(f'{self.image_save_dir}/diff_{name}.png'))

            del out, diff
        
        del pred, image
        torch.cuda.empty_cache()

    def on_validation_epoch_end(self) -> None:
        tb_logger = self.loggers[1].experiment
        _log_dict = {
            key.replace('step_', ''): torch.tensor(value).mean()
            for (key, value) in self.valid_step_outputs.items()
        }
        self.log_dict({'val_rec_loss': _log_dict['rec_loss']}, prog_bar=True)
        tb_logger.add_scalar('Loss/rec_loss', _log_dict['rec_loss'], self.current_epoch)

        _log_dict_metrics = {key.replace('metrics_', ''): value for (key, value) in _log_dict.items() if 'metrics' in key}
        self.log_dict(_log_dict_metrics, prog_bar=True)
        for k,v in _log_dict_metrics.items():
            tb_logger.add_scalar(f'Metrics/{k}'.replace('val_', ''), v, self.current_epoch)

        self.valid_step_outputs.clear()
    
    def on_predict_start(self):
        super().on_predict_start()
        self.train()

    def predict_step(self, batch: Dict[str, Tensor | str], batch_idx: int) -> None:
        with torch.no_grad():
            pred, input = self._val_step(batch, 0)
        
        min, max = batch['min'][0], batch['max'][0]
        denorm = self.denorm(min, max)
        pred = denorm(pred)
        input = denorm(input)

        anomaly_map = self.create_anomaly_map(pred, input)
        pred = pred.cpu().data.numpy()
        input = input.cpu().data.numpy()
        anomaly_map = anomaly_map.cpu().data.numpy()

        name, ext = self.get_name_ext(batch['filepath'][0], add_epoch=False)
        np.save(Path(f'{self.image_save_dir}/pred_{name}{ext}'), pred)
        np.save(Path(f'{self.image_save_dir}/anomaly_{name}{ext}'), anomaly_map)

        anomaly_map = Image.fromarray((anomaly_map * 255).astype(np.uint8))
        anomaly_map.save(Path(f'{self.image_save_dir}/anomaly_{name}.png'))
        
        del pred
        torch.cuda.empty_cache()
        

class VAEModule(BaseModel):
    def __init__(self, opt: Namespace, image_out_dir: str) -> None:
        super().__init__(opt, image_out_dir)

        self.model = VanillaVAE2(in_channels=opt.recon_in_channels, latent_dim=opt.recon_latent_size, input_size=opt.recon_patch_size)
        self.reconstruction_loss = MSELoss()
        self.kld_loss = kullback_leibler_divergence_loss
        
        self.train_step_outputs = {}
        self.valid_step_outputs = {}

        self.metrics = MetricCollection({
            'psnr': PeakSignalNoiseRatio(),
            'ssim': StructuralSimilarityIndexMeasure()
        })
        
    def forward(self, x: Tensor) -> Sequence[Tensor]:
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.opt.recon_lr_ae)
        return optimizer
    
    def get_current_beta(self) -> float:
        if self.opt.recon_beta_proportion > 1 or self.opt.recon_beta_proportion < 0:
            raise ValueError("R must be between 0 and 1")
        
        # Warmup
        if self.current_epoch < self.opt.recon_beta_warmup_epochs:
            return self.opt.recon_beta_start

        # Cyclical annealing
        cycle_size = self.total_beta_iterations // 4
        beta_step = self.global_step - self.opt.recon_beta_warmup_epochs * self.trainer.num_training_batches
        tau = (beta_step % cycle_size) / cycle_size
        if tau < self.opt.recon_beta_proportion:
            return self.opt.recon_beta_start + (
                self.opt.recon_beta_end - self.opt.recon_beta_start
            ) * (tau / self.opt.recon_beta_proportion)

        return self.opt.recon_beta_end
    
    def on_fit_start(self):
        check_path(self.image_save_dir)
    
    def on_train_start(self):
        if self.trainer.max_epochs < 20:
            raise ValueError("Training epochs should be at least 20")
        self.max_beta_epochs = self.trainer.max_epochs - 5
        self.total_beta_iterations = (
            self.max_beta_epochs - self.opt.recon_beta_warmup_epochs
        ) * self.trainer.num_training_batches
    
    def training_step(self, batch: Dict[str, str | Tensor], batch_idx: int) -> Tensor:
        image = batch['image']

        reconstruction, mu, log_var = self(image)
        beta = self.get_current_beta()
        reconstruction_loss = self.reconstruction_loss(reconstruction, image)
        kld_loss = self.kld_loss(mu, log_var, 1e-4)
        loss = reconstruction_loss + beta * kld_loss
        metrics = self.metrics(reconstruction.detach(), image)

        if self.global_step % (self.trainer.num_training_batches * 25) == 0:
            self.log_image("Input", (image * 255).to(torch.uint8))
            self.log_image("Reconstruction", (reconstruction * 255).to(torch.uint8))
            self.log_image("Difference", ((reconstruction - image) * 255).to(torch.uint8))

        if not self.train_step_outputs:
            self.train_step_outputs = {
                "step_loss": [loss],
                "step_rec_loss": [reconstruction_loss],
                "step_kld_loss": [kld_loss],
                "step_beta": [beta],
                "step_metrics_psnr": [metrics['psnr']],
                "step_metrics_ssim": [metrics['ssim']]
            }
        else:
            self.train_step_outputs["step_loss"].append(loss)
            self.train_step_outputs["step_rec_loss"].append(reconstruction_loss)
            self.train_step_outputs["step_kld_loss"].append(kld_loss)
            self.train_step_outputs["step_beta"].append(beta)
            self.train_step_outputs["step_metrics_psnr"].append(metrics['psnr'])
            self.train_step_outputs["step_metrics_ssim"].append(metrics['ssim'])
            
        return loss
    
    def on_train_epoch_end(self) -> None:
        tb_logger = self.loggers[0].experiment
        _log_dict = {
            key.replace('step_', ''): torch.tensor(value).mean()
            for (key, value) in self.train_step_outputs.items()
        }
        tb_logger.add_scalar('Beta', _log_dict['beta'], self.current_epoch)

        _log_dict_loss = {f'Loss/{key}': value for (key, value) in _log_dict.items() if 'loss' in key}
        for k,v in _log_dict_loss.items():
            tb_logger.add_scalar(k, v, self.current_epoch)

        _log_dict_metrics = {f'Metrics/{key}'.replace('metrics_', ''): value for (key, value) in _log_dict.items() if 'metrics' in key}
        for k,v in _log_dict_metrics.items():
            tb_logger.add_scalar(k, v, self.current_epoch)

        self.train_step_outputs.clear()
        
    def _val_step(self, batch: Dict[str, Tensor | str], loop_idx: int) -> Tensor:
        image = batch['image'][loop_idx]
        _, h, w = image.shape
        denorm = self.denorm(batch['min'][loop_idx], batch['max'][loop_idx])

        pred = torch.zeros_like(image, device=self.device)
        patch_overlap_count = torch.zeros_like(pred, device=self.device)

        if h == self.opt.recon_patch_size:
            x_range = list(np.array([0]))
        else:
            x_range = list(range(0, h - self.opt.recon_patch_size, self.opt.recon_stride))
            if (x_range[-1] + self.opt.recon_patch_size) < h :
                x_range.extend(range(h - self.opt.recon_patch_size, h - self.opt.recon_patch_size + 1))
        
        if w == self.opt.recon_patch_size:
            y_range = list(np.array([0]))
        else:
            y_range = list(range(0, w - self.opt.recon_patch_size, self.opt.recon_stride))
            if (y_range[-1] + self.opt.recon_patch_size) < w:
                y_range.extend(range(w - self.opt.recon_patch_size, w - self.opt.recon_patch_size + 1))

        for x in x_range:
            for y in y_range:
                patch = image[:, x : x + self.opt.recon_patch_size, y : y + self.opt.recon_patch_size]
                patch = self(patch.unsqueeze(0))[0].squeeze(0)
                pred[:, x : x + self.opt.recon_patch_size, y : y + self.opt.recon_patch_size] += patch
                
                patch_overlap_count[:, x : x + self.opt.recon_patch_size, y : y + self.opt.recon_patch_size] += torch.ones_like(patch, device=self.device)
        
        pred = torch.div(pred, patch_overlap_count)
        del patch_overlap_count
        pred = denorm(pred)
        image = denorm(image)
        torch.cuda.empty_cache()
        
        return pred, image
    
    def validation_step(self, batch: Dict[str, Tensor | str], batch_idx: int) -> None:
        pred, image = self._val_step(batch, 0)
        reconstruction_loss = self.reconstruction_loss(pred, image)
        pred = pred.cpu()
        image = image.cpu()
        val_metrics = MetricCollection({
            'psnr': PeakSignalNoiseRatio(),
            'ssim': StructuralSimilarityIndexMeasure()
        })
        metrics = val_metrics(pred.unsqueeze(0), image.unsqueeze(0))

        if not self.valid_step_outputs:
            self.valid_step_outputs = {
                "step_rec_loss": [reconstruction_loss],
                "step_val_metrics_psnr": [metrics['psnr']],
                "step_val_metrics_ssim": [metrics['ssim']]
            }
        else:
            self.valid_step_outputs["step_rec_loss"].append(reconstruction_loss)
            self.valid_step_outputs["step_metrics_psnr"].append(metrics['psnr'])
            self.valid_step_outputs["step_metrics_ssim"].append(metrics['ssim'])

        if self.global_step % (self.trainer.num_training_batches * 50) == 0:
            pred = pred.numpy()
            image = image.numpy()

            name, ext = self.get_name_ext(batch['filepath'][0])
            np.save(Path(f'{self.image_save_dir}/out_{name}.npy'), pred)
            out = Image.fromarray(equalize(pred.transpose(1, 2, 0).squeeze(), plower=0, pupper=100))
            out.save(Path(f'{self.image_save_dir}/out_{name}.png'))
            
            diff = Image.fromarray(equalize((image - pred).transpose(1, 2, 0).squeeze(), plower=0, pupper=100))
            diff.save(Path(f'{self.image_save_dir}/diff_{name}.png'))

            del out, diff
        
        del pred, image
        torch.cuda.empty_cache()

    def on_validation_epoch_end(self) -> None:
        tb_logger = self.loggers[1].experiment
        _log_dict = {
            key.replace('step_', ''): torch.tensor(value).mean()
            for (key, value) in self.valid_step_outputs.items()
        }
        self.log_dict({'val_rec_loss': _log_dict['rec_loss']}, prog_bar=True)
        tb_logger.add_scalar('Loss/rec_loss', _log_dict['rec_loss'], self.current_epoch)

        _log_dict_metrics = {key.replace('metrics_', ''): value for (key, value) in _log_dict.items() if 'metrics' in key}
        self.log_dict(_log_dict_metrics, prog_bar=True)
        for k,v in _log_dict_metrics.items():
            tb_logger.add_scalar(f'Metrics/{k}'.replace('val_', ''), v, self.current_epoch)

        self.valid_step_outputs.clear()
    
    def predict_step(self, batch: Dict[str, Tensor | str], batch_idx: int) -> None:
        pred, input = self._val_step(batch, 0)
        
        anomaly_map = self.create_anomaly_map(pred, input)
        anomaly_map = anomaly_map.cpu().data.numpy()
        pred = pred.cpu().data.numpy()

        name, ext = self.get_name_ext(batch['filepath'][0], add_epoch=False)
        np.save(Path(f'{self.image_save_dir}/pred_{name}{ext}'), pred)
        np.save(Path(f'{self.image_save_dir}/anomaly_{name}{ext}'), anomaly_map)
        anomaly_map = Image.fromarray((anomaly_map * 255).astype(np.uint8))
        anomaly_map.save(Path(f'{self.image_save_dir}/anomaly_{name}.png'))
        
        del pred, anomaly_map, input
        torch.cuda.empty_cache()


class MERLINModule(BaseModel):
    def __init__(self, opt: Namespace, image_out_dir: str) -> None:
        super().__init__(opt, image_out_dir)
        self.model = UNet_v2()

        self.loss = MSELoss()

        self.train_step_output = []

        self.valid_patch_size = 256
        self.valid_stride = 64
        
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def configure_optimizers(self) -> Dict[str, torch.optim.Optimizer | torch.optim.lr_scheduler.LRScheduler]:
        optimizer = Adam(params=self.parameters(), lr=self.opt.despeckler_lr)
        return {
            'optimizer': optimizer,
            'lr_scheduler': MultiStepLR(optimizer, milestones=[5, 20])
        }
        
    def on_fit_start(self):
        check_path(self.image_save_dir)
    
    def training_step(self, batch: Dict[str, Tensor | str], batch_idx: int) -> Tensor:
        input, target = batch['real'], batch['imag']
        if np.random.uniform() > 0.5:
            input, target = batch['imag'], batch['real']

        loss = self.loss(self(input), target)
        self.log('step_loss', loss, prog_bar=True)
        self.train_step_output.append(loss)

        return loss
    
    def on_train_epoch_end(self) -> None:
        loss = torch.tensor(self.train_step_output).mean()
        self.log('loss', loss)
        self.logger.experiment.add_scalar('loss', loss, self.current_epoch)
        self.train_step_output.clear()
        
    def _val_pred_step(self, batch: Dict[str, Tensor | str], loop_idx: int) -> Tensor:
        h, w = batch['real'][loop_idx].shape
        denorm = self.denorm(batch['min'][loop_idx], batch['max'][loop_idx])

        pred_real = torch.zeros(h, w, device=self.device)
        pred_imag = torch.zeros(h, w, device=self.device)

        patch_overlap_count = torch.zeros(h, w, device=self.device)

        if h == self.valid_patch_size:
            x_range = list(np.array([0]))
        else:
            x_range = list(range(0, h - self.valid_patch_size, self.valid_stride))
            if (x_range[-1] + self.valid_patch_size) < h :
                x_range.extend(range(h - self.valid_patch_size, h - self.valid_patch_size + 1))
        
        if w == self.valid_patch_size:
            y_range = list(np.array([0]))
        else:
            y_range = list(range(0, w - self.valid_patch_size, self.valid_stride))
            if (y_range[-1] + self.valid_patch_size) < w:
                y_range.extend(range(w - self.valid_patch_size, w - self.valid_patch_size + 1))

        for x in x_range:
            for y in y_range:
                real_patch = batch['real'][loop_idx][x : x + self.valid_patch_size, y : y + self.valid_patch_size]
                imag_patch = batch['imag'][loop_idx][x : x + self.valid_patch_size, y : y + self.valid_patch_size]
                
                real_patch, imag_patch = torch_symetrisation_patch(real_patch, imag_patch)

                real_patch = self(real_patch.unsqueeze(0).unsqueeze(0))
                pred_real[x : x + self.valid_patch_size, y : y + self.valid_patch_size] += real_patch.squeeze()

                imag_patch = self(imag_patch.unsqueeze(0).unsqueeze(0))
                pred_imag[x : x + self.valid_patch_size, y : y + self.valid_patch_size] += imag_patch.squeeze()

                patch_overlap_count[x : x + self.valid_patch_size, y : y + self.valid_patch_size] += torch.ones(self.valid_patch_size, self.valid_patch_size, device=self.device)
        
        pred_real = torch.div(pred_real, patch_overlap_count)
        pred_imag = torch.div(pred_imag, patch_overlap_count)
        del patch_overlap_count
        torch.cuda.empty_cache()
        
        pred_real = denorm(pred_real).cpu().data.numpy()
        pred_imag = denorm(pred_imag).cpu().data.numpy()
            
        return pred_real + 1j * pred_imag
    
    def validation_step(self, batch: Dict[str, Tensor | str], batch_idx: int) -> None:
        pred_cpx = self._val_pred_step(batch, 0)

        name, ext = self.get_name_ext(batch['filepath'][0])
        image = Image.fromarray(equalize(pred_cpx))
        image.save(Path(f'{self.image_save_dir}/{name}.png'))

        np.save(Path(f'{self.image_save_dir}/{name}{ext}'), pred_cpx)
    
    def on_predict_start(self):
        self.valid_stride = 32
            
    def predict_step(self, batch: Dict[str, Tensor | str], batch_idx: int) -> None:
        pred = self._val_pred_step(batch, 0)
        phase = batch['image_phase'][0].cpu().data.numpy()
        pred = np.abs(pred) * np.exp(1j * phase)
        
        name, ext = self.get_name_ext(batch['filepath'][0], add_epoch=False)
        np.save(Path(f'{self.image_save_dir}/{name}{ext}'), pred)
        
        del pred, phase
        torch.cuda.empty_cache()
