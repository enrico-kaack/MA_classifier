import d6tflow, luigi
from  tasks.preprocessing import TaskRuleProcessor, TaskVocabCreator
import logging

import enum
class ProblemType(enum.Enum):
    RETURN_NONE = "RETURN_NULL"

from models.random_forst import process_general_data

@d6tflow.inherits(TaskRuleProcessor, TaskVocabCreator)
class TaskPrepareXY(d6tflow.tasks.TaskPickle):
    problem_type = luigi.EnumParameter(enum=ProblemType)
    window_size = luigi.IntParameter(default=20)
    step_size = luigi.IntParameter(default=3)

    def requires(self):
        return {"data": self.clone(TaskRuleProcessor), "vocab": self.clone(TaskVocabCreator)}

    def run(self):
        data, vocab = self.inputLoad()

        x,y = process_general_data(data, vocab, window_size=self.window_size, step_size=self.step_size, problem_type=self.problem_type.value)

        self.save((x,y))


from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

@d6tflow.inherits(TaskPrepareXY)
class TaskTrainTestSplit(d6tflow.tasks.TaskPickle):
    test_split_percentage = luigi.FloatParameter(default=0.25)
    oversampling_enabled = luigi.BoolParameter(default=True)
    ratio_after_oversampling = luigi.FloatParameter(default=0.5)

    persist=['X_train','y_train', "X_test", "y_test"]

    def requires(self):
        return self.clone(TaskPrepareXY)

    def run(self):
        x,y = self.input().load()

        X_train, X_test, y_train, y_test  = train_test_split(x, y, test_size=self.test_split_percentage, random_state=1)

        if self.oversampling_enabled:
            oversample = RandomOverSampler(sampling_strategy=self.ratio_after_oversampling, random_state=1)
            X_train, y_train = oversample.fit_resample(X_train, y_train)

        logging.info(f"Length Train: {len(X_train)}, length Test {len(X_test)}")
        self.save({
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test
        })

from sklearn.ensemble import RandomForestClassifier
from collections import Counter

@d6tflow.inherits(TaskTrainTestSplit)
class TaskTrainRandomForest(d6tflow.tasks.TaskPickle):
    n_trees_in_forest = luigi.IntParameter(default=100)
    max_features = luigi.Parameter(default="sqrt")

    def requires(self):
        return self.clone(TaskTrainTestSplit)

    def run(self):
        X_train = self.input()["X_train"].load()
        y_train = self.input()["y_train"].load()
        X_test = self.input()["X_test"].load()
        y_test = self.input()["y_test"].load()

        train_counter = Counter(y_train)
        test_counter = Counter(y_test)
        print(train_counter)
        print(f"Feature Distribution: Train: {train_counter[1] *100/ len(y_train)}%, Test: {test_counter[1] *100/ len(y_test)}%")

        model = RandomForestClassifier(n_estimators=self.n_trees_in_forest, 
                                    random_state=1, 
                                    max_features = self.max_features,
                                    n_jobs=-1, verbose = True)

        model.fit(X_train, y_train)
        self.save(model)


from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, accuracy_score
import numpy as np
from utils.plotter import confusion_matrix, evaluate_model
from utils.plotter import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

@d6tflow.inherits(TaskTrainRandomForest, TaskTrainTestSplit)
class TaskEvaluateRandomForest(d6tflow.tasks.TaskPickle):

    def requires(self):
        return{"model": self.clone(TaskTrainRandomForest), "data": self.clone(TaskTrainTestSplit)}

    def run(self):
        model = self.input()["model"].load()
        X_train = self.input()["data"]["X_train"].load()
        y_train = self.input()["data"]["y_train"].load()
        X_test = self.input()["data"]["X_test"].load()
        y_test = self.input()["data"]["y_test"].load()

        # Training predictions (to demonstrate overfitting)
        train_rf_predictions = model.predict(X_train)
        train_rf_probs = model.predict_proba(X_train)[:, 1]

        # Testing predictions (to determine performance)
        rf_predictions = model.predict(X_test)
        rf_probs = model.predict_proba(X_test)[:, 1]


        n_nodes = []
        max_depths = []

        # Stats about the trees in random forest
        for ind_tree in model.estimators_:
            n_nodes.append(ind_tree.tree_.node_count)
            max_depths.append(ind_tree.tree_.max_depth)
            
        print(f'Average number of nodes {int(np.mean(n_nodes))}')
        print(f'Average maximum depth {int(np.mean(max_depths))}')



        # Plot formatting
        plt.style.use('fivethirtyeight')
        plt.rcParams['font.size'] = 18


        evaluate_model(rf_predictions, rf_probs, y_test,  train_rf_predictions, train_rf_probs, y_train)


        # Confusion matrix
        cm = confusion_matrix(y_test, rf_predictions)
        plot_confusion_matrix(cm, classes = ['0', '1'],
                            title = 'Confusion Matrix', normalize=True)