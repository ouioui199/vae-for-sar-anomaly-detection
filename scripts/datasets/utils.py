from typing import Sequence
import os

import numpy as np
from scipy import signal


def ensure_chw_format(x: np.ndarray) -> np.ndarray:
    """Ensure image is in CHW format, convert if necessary.
    
    Args:
        x (np.ndarray): Input image to check/convert format
        
    Returns:
        np.ndarray: Image in CHW format
        
    Raises:
        TypeError: If input is not numpy array or torch tensor
        ValueError: If input is not a 3D array
        
    Example:
        >>> img = np.zeros((64, 64, 3))  # HWC format
        >>> chw_img = ensure_chw_format(img)  # Converts to (3, 64, 64)
    """
    if len(x.shape) != 3:
        raise ValueError("Image must be 3D array")
    # Convert from HWC to CHW, channel is often the smallest dimension in a SAR image.
    if min(x.shape) != x.shape[0]:
        return x.transpose(2, 0, 1)
    return x


def is_directory_empty(directory_path: str) -> bool:
    """Check if a directory is empty.
    
    Args:
        directory_path: Path to the directory to check
        
    Returns:
        True if the directory is empty or doesn't exist, False otherwise
    """
    if not os.path.exists(directory_path):
        return True
        
    if os.path.isdir(directory_path):
        # Check if directory contains any files or subdirectories
        return len(os.listdir(directory_path)) == 0
    
    # If it's not a directory, return False
    return False


def symetrisation_patch(real_part: np.ndarray, imag_part: np.ndarray) -> Sequence[np.ndarray]:
    # FFT of the image, shift the 0 frequency value to the center of the spectrum
    S = np.fft.fftshift(np.fft.fft2(real_part + 1j * imag_part))
    # Azimuth 1D profile (averaging range)
    p = np.array([np.mean(np.abs(S[i, :])) for i in range(S.shape[0])])
    # Symetric profile
    sp = p[::-1]
    
    c = np.fft.ifft(np.fft.fft(p) * np.conjugate(np.fft.fft(sp))).real
    d1 = np.unravel_index(c.argmax(), p.shape[0])
    d1 = d1[0]
    
    shift_az_1 = int(round(-(d1 - 1) / 2)) % p.shape[0] + int(p.shape[0] / 2)
    p2_1 = np.roll(p, shift_az_1)
    
    shift_az_2 = int(round(-(d1 - 1 - p.shape[0]) / 2)) % p.shape[0] + int(p.shape[0] / 2)
    p2_2 = np.roll(p, shift_az_2)
    
    window = signal.windows.gaussian(p.shape[0], std=0.2 * p.shape[0])
    test_1 = np.sum(window * p2_1)
    test_2 = np.sum(window * p2_2)
    # make sure the spectrum is symetrized and zeo-Doppler centered
    if test_1 >= test_2:
        p2 = p2_1
        shift_az = shift_az_1 / p.shape[0]
    else:
        p2 = p2_2
        shift_az = shift_az_2 / p.shape[0]
    S2 = np.roll(S, int(shift_az * p.shape[0]), axis=0)

    # Compute range 1D profile by averaging azimuth
    q = np.array([np.mean(np.abs(S[:, j])) for j in range(S.shape[1])])
    sq = q[::-1]
    
    #correlation
    cq = np.fft.ifft(np.fft.fft(q) * np.conjugate(np.fft.fft(sq))).real
    d2 = np.unravel_index(cq.argmax(), q.shape[0])
    d2 = d2[0]
    
    shift_range_1 = int(round(-(d2 - 1) / 2)) % q.shape[0] + int(q.shape[0] / 2)
    q2_1 = np.roll(q,shift_range_1)
    
    shift_range_2 = int(round(-(d2 - 1 - q.shape[0]) / 2)) % q.shape[0] + int(q.shape[0] / 2)
    q2_2 = np.roll(q, shift_range_2)
    
    window_r = signal.windows.gaussian(q.shape[0], std=0.2 * q.shape[0])
    test_1 = np.sum(window_r * q2_1)
    test_2 = np.sum(window_r * q2_2)
    if test_1 >= test_2:
        q2 = q2_1
        shift_range = shift_range_1 / q.shape[0]
    else:
        q2 = q2_2
        shift_range = shift_range_2 / q.shape[0]

    Sf = np.roll(S2, int(shift_range * q.shape[0]), axis=1)
    # plot_im(np.abs(Sf),title='sym')
    ima2 = np.fft.ifft2(np.fft.ifftshift(Sf))
    # disp_sar(np.abs(ima2))
    return ima2.real, ima2.imag
