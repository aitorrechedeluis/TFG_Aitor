# TFG_Aitor

This repository contains the code to compute cell lipotypes localization in brain tissues by mass spectrometry. You will find both "lineal_model" and "neura_model" programs, each of it containing all the code needed to run.

> [!WARNING]
> Data from MALDI-MSI is not available.

## Usage

#### Lineal model:
Run main.py and images will be automatically generated. Program is set to generate "Brain_positive.imzML" imagenes. Other options:

    - "--input spectra": Change spectra in order to generate other BTS images.
    - "--clusters number": Change number in order to use an arbitrary number of clusters to remove background. Default value is 3.
    - "--threshold value": Change value to use another threshold value when binning the spectra. Defaul value is 50000.

#### Neural model:
Both notebooks have instructions to change input and output parameters. Run Neural_creator.ipynb to create and save the model and run predict_BTS to make predictions. 


