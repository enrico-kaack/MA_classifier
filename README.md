
## Dataset Naming
> Important naming missmatch: in the code, the terms for test and validation set are used wrongly and does not comply to the meaning of the default train-validation-test dataset: The test dataset in this repo is the validation dataset in common terminology and the validation(holdout) set in this repo is the commonly known test or holdout set that has not been used during training or tuning. The naming of the dataset follows the following form:
> final_dataset is the training dataset
> final_test is the validation dataset during tuning
> final_validation is the holdout or test dataset used for the performance evaluation and code manipulation.