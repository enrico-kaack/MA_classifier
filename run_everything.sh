#!/bin/bash

python -m scripts.run_experiments.train_all_models final_dataset -n 4
python -m scripts.run_experiments.test_holdout final_validation -n 4
python -m scripts.run_experiments.test_manipulated_code final_validation -n 4