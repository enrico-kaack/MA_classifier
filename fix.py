import argparse, sys
from tasks.preprocessing import ProblemType, TaskTrainTestSplit
from tasks.gradient_boosting_classifier import TaskTrainGradientBoostingClassifier
from tasks.svm import TaskTrainSVM
from tasks.lstm import TaskTrainLstm
from tasks.random_forest import TaskTrainRandomForest
from tasks.manipulate_code import TaskEvalEnsemble, TaskEvalKeras
import d6tflow

def run_all_tasks():
    task_list = []

    for problem_type in ProblemType:
        t =[ 
            TaskTrainTestSplit(problem_type=problem_type, oversampling_enabled=False, ratio_after_undersampling=0.1, undersampling_enabled=True, encode_type=True),
            TaskTrainTestSplit(problem_type=problem_type, oversampling_enabled=False, ratio_after_undersampling=0.5, undersampling_enabled=True, encode_type=True),
            TaskTrainTestSplit(problem_type=problem_type, oversampling_enabled=False, ratio_after_undersampling=0.1, undersampling_enabled=True, encode_type=True),
            TaskTrainTestSplit(problem_type=problem_type, oversampling_enabled=False, undersampling_enabled=False, encode_type=True),
            TaskTrainTestSplit(problem_type=problem_type, oversampling_enabled=False, undersampling_enabled=False, encode_type=False),
            TaskTrainTestSplit(problem_type=problem_type, oversampling_enabled=True, ratio_after_oversampling=0.5, undersampling_enabled=False, encode_type=False),
            TaskTrainTestSplit(problem_type=problem_type, oversampling_enabled=True, ratio_after_oversampling=0.5, undersampling_enabled=False, encode_type=True),
            TaskTrainTestSplit(problem_type=problem_type, oversampling_enabled=True, ratio_after_oversampling=1.0, undersampling_enabled=False, encode_type=False),
            TaskTrainTestSplit(problem_type=problem_type, oversampling_enabled=True, ratio_after_oversampling=1.0, undersampling_enabled=False, encode_type=True),
            TaskTrainTestSplit(problem_type=problem_type, oversampling_enabled=False, ratio_after_undersampling=0.5, undersampling_enabled=True, encode_type=False),
            TaskTrainTestSplit(problem_type=problem_type, oversampling_enabled=False, ratio_after_undersampling=0.5, undersampling_enabled=True, encode_type=True),
        ]
        task_list.extend(t)

    d6tflow.preview(task_list)
    d6tflow.run(task_list, workers=4)

if __name__ == "__main__":
    run_all_tasks()





