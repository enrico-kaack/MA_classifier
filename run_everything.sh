#!/bin/bash

python -m scripts.run_experiments.train_all_models final_dataset -n 4
#move generated json files in results in seperate folder
mkdir results/train_val
mv results/*.json results/train_val

python -m scripts.run_experiments.test_holdout final_validation -n 4
mkdir results/holdout_test
mv results/*.json results/holdout_test

python -m scripts.run_experiments.test_manipulated_code final_validation -n 4
mkdir results/manipulated_code
mv results/*.json results/manipulated_code