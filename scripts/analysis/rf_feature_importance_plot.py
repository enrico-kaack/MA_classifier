from tasks.preprocessing import ProblemType
from tasks.random_forest import TaskTrainRandomForest
from tasks.holdout_test import TaskPrepareXYHoldout
from sklearn.inspection import permutation_importance
import numpy as np
import matplotlib.pyplot as plt

def run():
    for problem_type in ProblemType:
        t = TaskTrainRandomForest(problem_type=problem_type, oversampling_enabled=True, ratio_after_oversampling=0.5, undersampling_enabled=False, encode_type=True)
        model = t.outputLoad()

        data_task = TaskPrepareXYHoldout(encode_type=True, problem_type=problem_type)
        x,y = data_task.outputLoad()

        result = permutation_importance(model, x, y, n_repeats=10,
                                        random_state=1, n_jobs=4)
        sorted_idx = result.importances_mean.argsort()

        fig, ax = plt.subplots()
        ax.boxplot(result.importances[sorted_idx].T,
                vert=False, labels=sorted_idx)
        ax.set_title("Permutation Importances (holdout set)")
        fig.tight_layout()
        plt.savefig(f"Importance_permuted{problem_type.value}.pdf")






if __name__ == "__main__":
    run()