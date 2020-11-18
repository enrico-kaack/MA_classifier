from tasks.preprocessing import ProblemType
from tasks.random_forest import TaskTrainRandomForest
from tasks.holdout_test import TaskPrepareXYHoldout
from sklearn.inspection import permutation_importance
import numpy as np
import matplotlib.pyplot as plt
import pickle

#first, use the run funtion to generate the importance_permutation data
#second, use the print_graph function to generate the plot
def run():
    for problem_type in ProblemType:
        t = TaskTrainRandomForest(problem_type=problem_type, oversampling_enabled=True, ratio_after_oversampling=0.5, undersampling_enabled=False, encode_type=True)
        model = t.outputLoad()

        data_task = TaskPrepareXYHoldout(encode_type=True, problem_type=problem_type)
        x,y = data_task.outputLoad()

        result = permutation_importance(model, x, y, n_repeats=10,scoring="f1",
                                        random_state=1, n_jobs=4)
        sorted_idx = result.importances_mean.argsort()

        with open(f"{problem_type}_rf_feature_importance.pickle", "wb") as out:
            pickle.dump((result, sorted_idx), out)


def print_graph(name, result, sorted_idx):
    fig, ax = plt.subplots()
    ax.boxplot(result.importances[sorted_idx].T,
            vert=False, labels=sorted_idx)
    ax.set_title("Permutation Importances")
    #fig.tight_layout()
    ax.set_xlabel("Decrease in f1 score")
    ax.set_ylabel("Label")

    plt.savefig(f"Importance_permuted{name}.pdf")






if __name__ == "__main__":
    for v in ["RETURN_NULL", "CONDITION_COMPARISON_SIMPLE", "CONDITION_COMPARISON"]:
        with open(f"results/FINAL/rf_feature_importance/{v}_rf_feature_importance.pickle", "rb") as p:
            data, sorted_idx = pickle.load(p)
            print_graph(v, data, sorted_idx)