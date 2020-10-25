import pandas as pd
import d6tflow

pd.set_option('display.max_colwidth', None)

from tasks.preprocessing import TaskTrainTestSplit, ProblemType

with open("size_resampling.txt", "w") as out:
    for problem_type in [ProblemType.RETURN_NONE, ProblemType.CONDITION_COMPARISON_SIMPLE, ProblemType.CONDITION_COMPARISON]:
        data = TaskTrainTestSplit(problem_type=problem_type, oversampling_enabled=False, ratio_after_oversampling = 0.5, undersampling_enabled=False, ratio_after_undersampling=0.5).output()["y_train"].load()
        out.write(f"No resampling: Type: {problem_type}: Size: {len(data)}\n")
        for v in [0.5, 1.0]:#oversampling
            data = TaskTrainTestSplit(problem_type=problem_type, oversampling_enabled=True, ratio_after_oversampling = v, undersampling_enabled=False, ratio_after_undersampling=0.5).output()["y_train"].load()
            out.write(f"Oversampling: {v}: Type: {problem_type}: Size: {len(data)}\n")
        for v in [0.1, 0.5]:#undersampling
            data = TaskTrainTestSplit(problem_type=problem_type, oversampling_enabled=False, ratio_after_oversampling = 0.5, undersampling_enabled=True, ratio_after_undersampling=v).output()["y_train"].load()
            out.write(f"Undersampling: {v}: Type: {problem_type}: Size: {len(data)}\n")
