
# Training Scripts for Master Thesis
This repository contains all scripts and data files for the machine learning part of the thesis "Automated Checking of Clean Code Guidelines for Python".

## Setup and Run

### One-Click Setup
The simplest way to run all training scripts is by using a docker container. To build and run the container, follow these steps:

1. Clone this repository and change into its directory.
2. Build the docker image
```
docker build . -t class
```

3. To run all train and test scripts, simply run the container. Use detached mode if required. Mount a folder for ``/classification/data`` (for the intermediate artifacts of the pipeline) and a folder for ``/classification/results`` for the results (create the local folders ``data`` and ``results`` if necessary):
```
docker run -v "$(pwd)"/data:/classification/data -v "$(pwd)"/results:/classification/results class
```
4. In order to execute other scripts (see below), login into the shell of the container:
```
docker run -it -v "$(pwd)"/data:/classification/data -v "$(pwd)"/results:/classification/results class /bin/bash
```

If you want to improve performance, you can increase the number of workers used in parallel in the ``run_everything.sh`` script. 

### Manual Setup
For manual setup, please consult the ``Dockerfile`` for instructions. Basically, after all dependencies are installed, the datasets have to be extracted into the project directory and the training scripts can be executed.

## Dataset Naming
> Important naming missmatch: in the code, the terms for test and validation set are used wrongly and do not comply to the general meaning of the train-validation-test dataset.

The test dataset in this repo is the validation dataset in common terminology and the validation(holdout) set in this repo is the commonly known test or holdout set that has not been used during training or tuning. 
 test dataset used for the performance evaluation and code manipulation.

| Dataset name | Usage |
|---|---|
| final_dataset | training dataset
| final_test | validation dataset during tuning
| final_validation | holdout or test dataset used for the performance evaluation and code manipulation
| final_dataset_complete | the full dataset before splitting into validation and test dat

### Location
The zip files of the dataset are located under ``datasets/``. For training, I recommend extracting the zip files into the project directory (like in the ``Dockerfile``).

## Project Overview

### Pipeline
The project uses the d6tflow framework to build the machine learning pipeline. The pipeline steps (called tasks) are defined in ``tasks/``. 

The ``preprocessing.py`` contains all preprocessing steps.

Then a python file for every model contains the training and validation tasks.

In ``holdout_test.py``, two tasks for testing a trained model on the test (holdout) dataset are defined. One for evalutating the ensemble-based models from scikit-learn and one for evaluating the Keras based LSTM.

In ``manipulate_code.py``, the necessary tasks for RQ3 are defined. This includes the code manipulation and the evaluation of trained models on this manipulated code (again divided into a task for the ensemble-based models from scikit-learn and for the Keras based LSTM).

The pipeline stores evaluation results as json files in the ``results``directory. We recommend moving the json files after training, after holdout evaluation and after code manipulation and evaluation into seperated folders to prevent a mixing (see the ``run_everything.sh`` script).

## Results
The results from our final tests are stored in ``results/FINAL``.
- The ``test`` directory contains the the training and validation results (see the remark about naming in the Dataset section of this document).
- The ``holdout_validation`` directory contains the results for the evaluation on the test (holdout) set.
- In ``man_code_final``, we stored the results for code manipulation in RQ3.
- In ``error_analysis``, we extracted true positives, false positives, true negatives and false negatives from different models and different configurations. To reference the exact training configuration, look up in the performance table in the thesis.
- The ``dataset_analysis`` dir contains analysis results from the datasets
- ``rf_feature_importance``contains data und images for the feature importance of the random forest classifer (best configuration)

To review the results, consult and adapt the ``misc_notebooks/analyse_results.ipynb`` for a pandas evaluation.

Older results are stored in the ``old`` folder.

## Data Downloader
The ``input folder`` contains a script to download repositories from Github (``data_downloader.py``). It requres a .txt file with organisation, repo name, branch and commit. In ``repos_final_with_commit_hash.txt``, we list all repositories we downloaded.
The files that we moved into the test and validation set are in the ``files_for_final_test|validation_extracted.txt`` (see the remark about the dataset naming in section dataset). The script ``extract_files_from_folders.py`` is responsible for extracting data.

## Scripts
The ``scripts`` folder contains several scripts used during this project. Consult this scripts and the documentation of d6tflow to learn how to work with the pipeline.

To run the scripts, use the following syntax in the commandline:
```
python -m scripts.run_experiments.train_all_models
```


### run_experiments
This folder contains the scripts to run the training and evaluation pipeline for RQ2 and RQ3. When executing the script without parameter, it will show a description of the parameters (or have a look inside the Python file). If you change the dataset names, you may have to provide it as parameters to the pipeline, since the names we use are the default values inside the pipeline classes. Consult the pipeline classes in the ``tasks`` folder for the parameter name.

### latex
This folder contains two scripts to generate the latex files for the appendix tables containing all results for RQ2 (``generate_train_result_tables.py``) and RQ3 (``generate_manipulate_code_result.py``). Remark: Again, the naming convention for the datasets is wrong as described in the dataset section.

### analysis
This section contains several scripts to analyse the dataset and resuls. See the comment at the top what purpose those scripts fullfil. Not all scripts are polished to contain an argument parser, the parameter for those have to be edited in the python file.
