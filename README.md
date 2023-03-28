# Slow Representation Learning
This repository contains the code for the experiments conducted in the paper [Autoencoding slow representations for semi-supervised data-efficient regression](https://link.springer.com/article/10.1007/s10994-022-06299-1).
                               
# Requirements
This repository requires a custom dataloader for hdf5 datasets. Check out the [parallel dataloader](https://github.com/Oleffa/parallel_dataloader) repository. The parallel dataloader is able to quickly load and cycle large image datasets through the memory of a GPU or CPU.
**Note:** there is also a miniaml example in the folder `./minimal_example` with the MNIST dataset that shows how to use the basic VAE tools in thes repository without the need for the above dataloader.
                               
# Datasets
The datasets used can be obtained from google drive:
- [Ball dataset](https://drive.google.com/drive/folders/1lCZQRBxDBypk6iWpZTE9zskWflTSyOy2?usp=sharing)
- [Pong dataset](https://drive.google.com/drive/folders/1Ci7wMvGSADEVEFEHsz9wF4pIkS-um9O6?usp=sharing) 
- [DeepMind Lab dataset](https://drive.google.com/drive/folders/16CfEsUmvN1g3-IKEeYkq9QiyRYPNuAXl?usp=sharing) 
Make sure to set the dataset directory in the appropriate experiment's vaeconfig.

# Usage
An experiment setup for a specific dataset is defined in a folder. In the following example I will us the included ball example defined in the experiment folder `./blob`. A setup must contain the following specifications:
- `./blob/vaeconfig.py`: This file contains hyperparameters that all experiments share such as dataset parameters, learning rates and settings for the training.
- `./blob/network/network.py`: contains the VAE architectur. The basic architecture is from this [beta-VAE pytorch implementation](https://github.com/1Konny/Beta-VAE) and has been adjusted to fit the different datasets. 
- `./blob/network/ds_Network.py`: contains the network architecture for the downstream task.
- `./blob/network/train_functions.py`: contains data preprocessing functions, specific loss function settings, houskeeping functions and functions to visualise the dataset.

An example to run an experiment for each dataset with one of the hyperparameter configurations used in the paper experiments is given in the file `./test.sh`

The results are stored in `./blob/models/`.
Following this setup one can easily setup experiments with other datasets by editing the `./experiment/vaeconfig.py` and the VAE architecture, data-preprocessing and downstream network architectures in `./experiment/network`.
The dataset has to be in hdf5 format. More details can be found in the [parallel dataloader](https://github.com/Oleffa/parallel_dataloader) repository

# Loading models
In this repository a model is more than just a torch module. Models are derived from a custom model class `./utils/model.py`. This class has many additional functions such as model loading/saving, hyperparameter storage, houskeeping storage and visualisation. Thus loading a model loads the entire pickled model class containing all the houskeeping data and model weights such that the model weights and results can be accessed easily.

# Other notes
## Minimal example
The minimal example in `./minimal_example` is a simplified version of the experiment setup if you just want to use one of the methods compared in the paper or want to check the implementations of the loss functions. It consists of a basic VAE setup without the need for a custom dataloader or a custom model class. It uses the MNIST dataset as an example and has no real experimental meaning beyond demonstration purpose as the dataset is not sequential.
## Loss functions
The general loss functions S-VAE, Slow-VAE, beta-VAE and L1/L2 loss are specified in `./network/train_functions_global.py` and used in the experiment specific loss functions.
