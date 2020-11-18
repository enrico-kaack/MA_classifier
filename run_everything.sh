#!/bin/bash

mkdir results/train_val
python -m scripts.run_experiments.train_all_models final_dataset -n 1
mv results/*.json results/train_val

mkdir results/holdout_test
python -m scripts.run_experiments.test_holdout final_validation -n 1
mv results/*.json results/holdout_test

mkdir results/manipulated_code
python -m scripts.run_experiments.test_manipulated_code final_validation -n 1
mv results/*.json results/manipulated_code