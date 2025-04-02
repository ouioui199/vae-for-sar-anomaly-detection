from typing import Sequence
from pathlib import Path
from argparse import ArgumentParser, Namespace
import glob, sys

from PIL import Image
import numpy as np
from tqdm import tqdm

sys.path.append('./scripts')
sys.path.append('./scripts/datasets')

from utils import ArgumentParsing, combine_polar_channels
from datasets.utils import ensure_chw_format


def local_reed_xiaoli_detector(
    image: np.ndarray,
    pixel_coordinates: Sequence[int],
    exclusion_window_size: int,
    box_car_size: int
) -> np.ndarray:
    # Ensure the image is in CHW format
    image = ensure_chw_format(image)
    c, h, w = image.shape

    # Extract test pixel
    x, y = pixel_coordinates
    test_pixel = image[:, x, y].reshape(c, 1)

    # Define background region boundaries
    half_box = box_car_size // 2
    half_excl = exclusion_window_size // 2

    # Define box car region
    x_start = x - half_box
    x_end = x + half_box + 1
    y_start = y - half_box
    y_end = y + half_box + 1    

    # Extract box car region
    box_car = image[:, x_start:x_end, y_start:y_end]

    # Create mask with 1s (include) and 0s (exclude)
    mask = np.ones((box_car_size, box_car_size), dtype=bool)

    # Set exclusion window to 0
    excl_start_x = half_box - half_excl
    excl_end_x = half_box + half_excl + 1
    excl_start_y = half_box - half_excl
    excl_end_y = half_box + half_excl + 1
    mask[excl_start_x:excl_end_x, excl_start_y:excl_end_y] = 0

    # Reshape box_car to (c, box_car_size*box_car_size)
    box_car_flat = box_car.reshape(c, -1)

    # Flatten mask and use it to filter background pixels
    mask_flat = mask.flatten()
    background = box_car_flat[:, mask_flat]

    # Compute background statistics
    if background.shape[1] <= c:  # Not enough samples
        return 0.0
    
    # # Compute mean and centered background
    # mu = np.mean(background, axis=1, keepdims=True)
    # centered_bg = background - mu

    # # Compute covariance with regularization
    # cov = (centered_bg @ np.conj(centered_bg.T)) / (background.shape[1] - 1)
    # cov += np.eye(c) * 1e-6  # Regularization

    cov = (background @ np.conj(background.T)) / (background.shape[1] - 1)
    cov += np.eye(c) * 1e-6  # Regularization

    # Compute RX detector
    try:
        inv_cov = np.linalg.inv(cov)
        # centered_test = test_pixel - mu
        # rx_value = (np.conj(centered_test.T) @ inv_cov @ centered_test)[0, 0].real
        rx_value = (np.conj(test_pixel.T) @ inv_cov @ test_pixel)[0, 0].real
        return rx_value
    except np.linalg.LinAlgError:
        return np.nan * np.ones_like(cov)     
    

def compute_reed_xiaoli_map(opt: Namespace, datadir: Path) -> None:
    paths = glob.glob(f'{datadir}/*.npy')
    path_rx = [p for p in paths if ('Combine' in p) and ('crop' in p)]
    if not path_rx:
        print('No combined polarization image found. Computing combined polarization image...')
        combine_polar_channels(paths)
        paths = glob.glob(f'{datadir}/*.npy')
        path_rx = [p for p in paths if 'Combine' in p]
    
    for p_rx in tqdm(path_rx):
        image = np.load(p_rx, mmap_mode='r')
        c, h, w = image.shape
        output = np.zeros((h, w))
        for i in range(opt.rx_box_car_size // 2, h - opt.rx_box_car_size // 2):
            for j in range(opt.rx_box_car_size // 2, w - opt.rx_box_car_size // 2):
                output[i, j] = local_reed_xiaoli_detector(image, (i, j), opt.rx_exclusion_window_size, opt.rx_box_car_size)
        
        output = output[np.newaxis, opt.rx_box_car_size // 2 : -opt.rx_box_car_size // 2, opt.rx_box_car_size // 2 : -opt.rx_box_car_size // 2]
        np.save(p_rx.replace('Combined', 'RX-map'), output)

        output = Image.fromarray(((output.squeeze() / np.max(output)) * 255).astype(np.uint8))
        output.save(p_rx.replace('Combined', 'RX-map').replace('.npy', '.png'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = ArgumentParsing(parser)
    opt = parser.parser.parse_args()

    datadir = Path(f'{opt.datadir}/{opt.data_band}_band/train/slc')
    compute_reed_xiaoli_map(opt, datadir)

    datadir = Path(f'{opt.datadir}/{opt.data_band}_band/predict/slc')
    compute_reed_xiaoli_map(opt, datadir)
