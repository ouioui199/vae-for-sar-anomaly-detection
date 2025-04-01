# Anomaly Detection in SAR imaging

## Abstract
In this paper, we propose an unsupervised learning approach for anomaly detection in SAR imaging. The proposed model combines a preprocessing despeckling step, a $\beta$-*annealing* Variational Auto-Encoder (VAE) for unsupervised anomaly filtering, and an anomaly detector based on the change of the covariance matrix at the input and output of the $\beta$-*annealing* VAE network. Experiments have been carried out on X-band ONERA polarimetric SAR images to demonstrate the effectiveness of Beta-Annealing VAE compared with the methods proposed in the literature.

## Architecture
![VAE architecture](images/VAE.png)

## Getting started
Anomaly Detection in SAR imaging with Adversarial AutoEncoder, Variational AutoEncoder and Reed-Xiaoli Detector.
To begin, clone the repository with ssh or https:

```
git clone git@gitlab-research.centralesupelec.fr:anomaly-detection-huy/aae_huy.git
git clone https://gitlab-research.centralesupelec.fr/anomaly-detection-huy/aae_huy.git
```

### Environment
Create a virtual environment with miniconda or other tools.
Details to install miniconda could be found [here](https://www.anaconda.com/docs/getting-started/miniconda/install).

### Install requirements
```
pip install -r requirements.txt
```

Install torchcvnn latest developments and install it as a library
```
git clone --single-branch --branch dev_transforms https://github.com/ouioui199/torchcvnn.git
pip install -e torchcvnn
```

We will use Pytorch-Lightning to organize our code. Documentations can be found [here](https://lightning.ai/docs/pytorch/stable/starter/introduction.html)

## Data folder structure
For quad-polarization images, the data folder container MUST be organized like below. Create folders if needed. Fell free to rename ```data_folder1``` and ```data_folder2``` to best organize your dataset.
```
|- data_folder1/
|   |- L_band/
|   |- UHF_band/
|   |- X_band/
|   |   |- train/
|   |   |- predict/
|   |   |   |- despeckled/
|   |   |   |- reconstructed/
|   |   |   |- slc/
|   |   |   |   |- something_Hh_something.npy
|   |   |   |   |- something_Hv_something.npy
|   |   |   |   |- something_Vh_something.npy
|   |   |   |   |- something_Vv_something.npy
|- data_folder2/
|   |- L_band/
|   |   |- train/
|   |   |- predict/
|   |   |   |- despeckled/
|   |   |   |- reconstructed/
|   |   |   |- slc/
|   |   |   |   |- something_Hh_something.npy
|   |   |   |   |- something_Hv_something.npy
|   |   |   |   |- something_Vh_something.npy
|   |   |   |   |- something_Vv_something.npy
|   |- UHF_band/
etc.
```
HvVh polarization should be pre-computed from Hv and Vh, with HvVh = (Hv + Vh) / 2. Normalization values will be computed automatically during the data processing, you don't need to do anything.

## Training
First, you need to train the despeckler. We use [MERLIN](https://ieeexplore.ieee.org/document/9617648) algorithms.
The code will outputs and save checkpoints to ```weights_storage/version_X/despeckler/*.ckpt```. Remember to change 'X' to your version. In the shell file has already been programmed to run sequentially 4 channels of a full polarization SAR image. If you wish to run it only on certain channel, comment the concerned code.
```
bash train_despeckler.sh > train_despeckler_log.txt 2>&1
```
The despeckler training has now finished, you must compute predictions to get despeckled SAR images.
```
bash predict_despeckler.sh > pred_despeckler_log.txt 2>&1
```
The code will predict sequentially on 4 channels of a full polarization SAR image, and store automatically despeckled images into specific folder. For example, if ```DATADIR=/your/data/directory/data_folder1/```, despeckled will be in ```/your/data/directory/data_folder1/train/despeckled``` folder.

#TODO here. Need to recheck how to store images and how images are moved in between. tensorboard

 It will need a despeckled image, so you need to precise arguments ```--despeckler_predict --despeckler_ckpt_path``` in the command. The code will run prediction and store the despeckled image into the **data** folder. Remember to change 'X' to your arguments.
```
python train_reconstructor.py --version X --aae_in_channels X --aae_epochs X --despeckler_predict --despeckler_ckpt_path <your_despeckler_ckpt_path> --despeckler_predict_datadir "directory1, directory2, etc"
```

For future runs, if you decide to keep using the same despeckled data, remove the two last arguments.
```
python train_reconstructor.py --version X --aae_in_channels X --aae_epochs X
```

If you decide to run the **Reconstructor** on specific data folder, run
```
python train_reconstructor.py --version X --aae_in_channels X --aae_epochs X --aae_datadir "directory"
```
The ```--aae_datadir``` argument does not accept multiple entries, so you need to store all your data into the desired folder.

To see all possible arguments, run
```
python train_despeckler.py --help
```

## Folder structure
Once start running the code, the folder will be organized as below. After cloning the code, create environments, install dependencies, you can start immediately the training. No further actions are required. All folders will be created automatically.
```
|- images/
|- scripts/
|   |- datasets/
|   |- models/
|   |- predict_despeckler.py
|   |- predict_reconstructor.py
|   |- train_despeckler.py
|   |- train_reconstructor.py
|   |- utils.py
|- training_logs/
|   |- version X/
|   |   |- despeckler
|   |   |   |- visualization
|   |   |   |- validation_samples
|   |   |- reconstructor
|   |   |   |- visualization
|   |   |   |- validation_samples
|- weights_storage/
|   |- version X/
|   |   |- despeckler
|   |   |- reconstructor
|- .gitignore
|- compute_RX.py
|- predict_despeckler.sh
|- predict_reconstructor.sh
|- README.md
|- requirements.txt
|- train_despeckler.sh
|- train_reconstructor.sh
```
