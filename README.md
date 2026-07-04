# AdaptFFSL-DS
The code implements AdaptFFSL-DS, an adaptive decision-making framework for Federated Few-Shot Learning. It uses ResFed as the local model along with a device-selection agent to choose devices based on their characteristics. The framework also adjusts the number of local epochs to balance accuracy and training speed.

## Requirements
AdaptFFSL-Ds has been implemented and tested with the following versions:

Python (v3.11.3).
Pytorch (v2.0.0).
Scikit-Learn (v1.2.2).
Scipy (v1.10.1).
FedLab (v1.3.0).
NumPy (v1.24.3).

## Installation
### Environment
Anaconda
Install anaconda for your platform

### Cuda Driver
You should be sure that you cuda driver has been installed or updated before.

### Create an Environment Variable and install packages
conda create -n your-env-name python=3.11

### Activate Environment
conda activate your-env-name

### Install Packages
 install requirements module
 
## Dataset
The dataset is hosted separately and is not included in this repository.

1. Download the dataset from:
   - https://drive.google.com/drive/folders/1HFBK0DwNzx32WUPz6yMfkKu8Y3NlHaSA
2. Extract the downloaded archive.
3. Move the extracted dataset to the `data/` directory.

## Initialize Clients
python train_lightFFSLDS_DDQN.py --make_dataset

## Train the Federated Engine
python train_lightFFSLDS_DDQN.py
