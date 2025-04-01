from typing import Sequence
from types import ModuleType
import os, glob, shutil

import torch
import numpy as np
from torch import Tensor
from scipy import signal


def check_path(path: str) -> None:
    """Check if a path exist. If not, create it. Else, remove all file and folders inside of it.

    Args:
        path (str): Path to check
    """
    
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        for file in glob.glob(os.path.join(path, '*')):
            if os.path.isdir(file):
                shutil.rmtree(file)
            else:
                os.remove(file)


def torch_symetrisation_patch(real_part: Tensor, imag_part: Tensor) -> Sequence[Tensor]:
    # FFT of the image, shift the 0 frequency value to the center of the spectrum
    S = torch.fft.fftshift(torch.fft.fft2(real_part + 1j * imag_part))
    # Range or Azimuth 1D profile (by averaging other dimension)
    p = torch.tensor([torch.mean(torch.abs(S[i, :])) for i in range(S.shape[0])])
    # Symetric profile
    sp = p.flip(dims=[0])
    
    c = torch.fft.ifft(torch.fft.fft(p) * torch.conj(torch.fft.fft(sp))).real
    d1 = torch.unravel_index(c.argmax(), p.shape[0])
    d1 = d1[0]
    
    shift_az_1 = int(torch.round(-(d1 - 1) / 2)) % p.shape[0] + int(p.shape[0] / 2)
    p2_1 = torch.roll(p, shift_az_1)
    
    shift_az_2 = int(torch.round(-(d1 - 1 - p.shape[0]) / 2)) % p.shape[0] + int(p.shape[0] / 2)
    p2_2 = torch.roll(p, shift_az_2)
    
    window = torch.tensor(signal.windows.gaussian(p.shape[0], std=0.2 * p.shape[0]))
    test_1 = torch.sum(window * p2_1)
    test_2 = torch.sum(window * p2_2)
    # make sure the spectrum is symetrized and zeo-Doppler centered
    if test_1 >= test_2:
        p2 = p2_1
        shift_az = shift_az_1 / p.shape[0]
    else:
        p2 = p2_2
        shift_az = shift_az_2 / p.shape[0]
    S2 = torch.roll(S, int(shift_az * p.shape[0]), dims=0)

    # Range or Azimuth 1D profile (by averaging other dimension)
    q = torch.tensor([torch.mean(torch.abs(S[:, j])) for j in range(S.shape[1])])
    sq = q.flip(dims=[0])
    
    #correlation
    cq = torch.fft.ifft(torch.fft.fft(q) * torch.conj(torch.fft.fft(sq))).real
    d2 = torch.unravel_index(cq.argmax(), q.shape[0])
    d2 = d2[0]
    
    shift_range_1 = int(torch.round(-(d2 - 1) / 2)) % q.shape[0] + int(q.shape[0] / 2)
    q2_1 = torch.roll(q,shift_range_1)
    
    shift_range_2 = int(torch.round(-(d2 - 1 - q.shape[0]) / 2)) % q.shape[0] + int(q.shape[0] / 2)
    q2_2 = torch.roll(q, shift_range_2)
    
    window_r = torch.tensor(signal.windows.gaussian(q.shape[0], std=0.2 * q.shape[0]))
    test_1 = torch.sum(window_r * q2_1)
    test_2 = torch.sum(window_r * q2_2)
    if test_1 >= test_2:
        q2 = q2_1
        shift_range = shift_range_1 / q.shape[0]
    else:
        q2 = q2_2
        shift_range = shift_range_2 / q.shape[0]

    Sf = torch.roll(S2, int(shift_range * q.shape[0]), dims=1)

    ima2 = torch.fft.ifft2(torch.fft.ifftshift(Sf))

    return ima2.real, ima2.imag


class MinMaxDenormalize:

    def __init__(self, min: np.ndarray | Tensor, max: np.ndarray | Tensor) -> None:
        self.min = min
        self.max = max
        
    def denorm_ndarray(self, norm_image: np.ndarray) -> np.ndarray:
        log_image = (self.max - self.min) * norm_image + self.min
        return np.exp(log_image) - np.spacing(1)
    
    def denorm_tensor(self, norm_image: Tensor) -> Tensor:
        log_image = (self.max - self.min) * norm_image + self.min
        return torch.exp(log_image )- np.spacing(1)

    def __call__(self, norm_image: np.ndarray | Tensor) -> np.ndarray | Tensor:
        if isinstance(norm_image, np.ndarray):
            return self.denorm_ndarray(norm_image)
        elif isinstance(norm_image, Tensor):
            return self.denorm_tensor(norm_image)


class LogDenormalize:

    def __init__(self, min_value: np.ndarray | Tensor, max_value: np.ndarray | Tensor, keep_phase: bool = False) -> None:
        self.min_value = min_value
        self.max_value = max_value
        self.keep_phase = keep_phase
        
    def log_denormalize_amplitude(
        self,
        backend: ModuleType,
        x: np.ndarray | Tensor,
    ) -> np.ndarray:
        # Handle complex input
        is_complex = "complex" in str(x.dtype)
        amplitude = backend.abs(x) if is_complex else x
        phase = backend.angle(x) if is_complex else None
            
        amplitude *= backend.log(self.max_value / self.min_value)
        amplitude = self.min_value * backend.exp(amplitude)
        
        return amplitude * backend.exp(1j * phase) if self.keep_phase else amplitude
    
    def denorm_ndarray(self, norm_image: np.ndarray) -> np.ndarray:
        return self.log_denormalize_amplitude(np, norm_image)

    def denorm_tensor(self, norm_image: Tensor) -> Tensor:
        return self.log_denormalize_amplitude(torch, norm_image)

    def __call__(self, norm_image: np.ndarray | Tensor) -> np.ndarray | Tensor:
        if isinstance(norm_image, np.ndarray):
            return self.denorm_ndarray(norm_image)
        elif isinstance(norm_image, Tensor):
            return self.denorm_tensor(norm_image)
