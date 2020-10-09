from tasks.preprocessing import ProblemType
from tasks.random_forest import TaskTrainRandomForest
import numpy as np
import matplotlib.pyplot as plt

def run():
    for problem_type in ProblemType:
        t = TaskTrainRandomForest(problem_type=problem_type, oversampling_enabled=True, ratio_after_oversampling=0.5, undersampling_enabled=False, encode_type=True)
        model = t.outputLoad()

        importances = model.feature_importances_
        indices = np.argsort(importances)


        plt.title('Feature Importances')
        plt.barh(range(len(indices)), importances[indices], color='#8f63f4', align='center')
        plt.yticks(range(len(indices)), indices)
        plt.xlabel('Relative Importance')
        plt.savefig(f"Importance_{problem_type.value}.pdf")






if __name__ == "__main__":
    run()