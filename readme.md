# EPILEPTIC SEIZURE DETECTION

## Introduction

This is a project for the course "MAPSIV" at UAB. The goal is to detect epileptic seizures in EEG signals. The project is divided in two parts:
1. Preprocessing of the EEG signals to obtain windows
2. Classification of the windows to detect seizures

## Conda environment

The conda environment is defined in the file `env.txt`. To create the environment, run the following command:

```bash
conda create --name <env> --file env.txt
```

## Preprocessing

The preprocessing is done by the object "DataHandler". It is defined in the file "dataHandler.py". To run the preprocessing, run the following command:

```bash
python main.py preprocess <path_of_raw_data> <path_to_store_window_data> <path_to_store_metadata>
```


