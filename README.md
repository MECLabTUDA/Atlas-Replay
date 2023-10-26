# Atlas-Replay -- Continual atlas-based segmentation (WACV 2024)

This repository represents the official PyTorch code base for our WACV 2024 published paper **Continual atlas-based segmentation of prostate MRI**. Our code exclusively utilizes the PyTorch version of the [VoxelMorph](https://github.com/voxelmorph/voxelmorph) framework as its foundation. For more details, please refer to our paper at [insert paper link].


## Table Of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [How to get started?](#how-to-get-started)
4. [Data, splits and pre-trained models](#data-splits-and-pre-trained-models)
5. [Citations](#citations)
6. [License](#license)

## Introduction

This WACV 2024 submission currently includes the following methods for Continual Learning:
* Sequential Training
* Rehearsal Training
* Riemannian Walk
* Elastic Weight Consolidation
* Bias Correction
* Incremental Learning Techniques

## Installation

The simplest way to install all dependencies is by using [Anaconda](https://conda.io/projects/conda/en/latest/index.html):

1. Create a Python 3.9 environment as `conda create -n <your_conda_env> python=3.9` and activate it as `conda activate  <your_conda_env>`.
2. Install CUDA and PyTorch through conda with the command specified by [PyTorch](https://pytorch.org/). The command for Linux was at the time `conda install pytorch torchvision cudatoolkit=11.3 -c pytorch`. Our code was last tested with version 1.13. Pytorch and TorchVision versions can be specified during the installation as `conda install pytorch==<X.X.X> torchvision==<X.X.X> cudatoolkit=<X.X> -c pytorch`. Note that the cudatoolkit version should be of the same major version as the CUDA version installed on the machine, e.g. when using CUDA 11.x one should install a cudatoolkit 11.x version, but not a cudatoolkit 10.x version.
3. Navigate to the project root (where `setup.py` lives).
4. Execute `pip install -r requirements.txt` to install all required packages.


## How to get started?
- Since our code base follows the VoxelMorph Framework, the models with Continual Learning methods are trained in the same fashion.
- If the VoxelMorph U-Net should be used for simple segmentation, please use the `--seg` flag, an example train script can be found [here](https://github.com/MECLabTUDA/Atlas-Replay/tree/main/scripts/torch/train_abstract_unet.py).
- The easiest way to start is using our `train_abstract_*.py` python files. For every baseline and Continual Learning method, we provide specific `train_abstract_*.py` python files, located in the [scripts folder](https://github.com/MECLabTUDA/Atlas-Replay/tree/main/scripts/torch).
- The [eval folder](https://github.com/MECLabTUDA/Atlas-Replay/tree/main/eval) contains several jupyter notebooks that were used to calculate performance metrics and plots used in our submission.


## Data, splits and pre-trained models
- **Data**: In our paper, we used seven publicly available prostate datasets where inter-subject samples are rigidly aligned using [SimpleITK](https://simpleitk.org/):
  - [Multi-site Dataset for Prostate MRI Segmentation](https://liuquande.github.io/SAML/)
  - [Prostate Dataset from the Medical Decathlon Challenge](https://drive.google.com/file/d/1Ff7c21UksxyT4JfETjaarmuKEjdqe1-a/view?usp=share_link)
- **Splits**: The train and test splits which were used during training for all methods can be found in the [misc folder](https://github.com/MECLabTUDA/Atlas-Replay/tree/main/misc) of this repository.
- **Models**: Our pre-trained models from our submission can be provided by contacting the [main author](mailto:amin.ranem@gris.informatik.tu-darmstadt.de) upon request.
- **Prototypes**: Our generated prototypes along with the preprocessed dataset can be requested [per mail](mailto:amin.ranem@gris.informatik.tu-darmstadt.de).

For more information about Atlas-Replay, please read the following paper:
```
Ranem, A., González, C., Pinto dos Santos, D., Othman, A., Bucher, A. & Mukhopadhyay, A. (2023).
Continual atlas-based segmentation of prostate MRI.
```

## Citations
If you are using Atlas-Replay or our code base for your article, please cite the following paper:
```
@article{ranem2023continual,
  title={Continual atlas-based segmentation of prostate MRI},
  author={Ranem, Amin and Gonz{\'a}lez, Camila and Pinto dos Santos,
          Daniel and Othman, Ahmed and Bucher, Andreas and Mukhopadhyay, Anirban}
}
```

## License

[Apache License 2.0](https://choosealicense.com/licenses/apache-2.0/)
