import d6tflow, luigi
from  tasks.preprocessing import TaskRuleProcessor, TaskVocabCreator, TaskPrepareXY, TaskTrainTestSplit
import logging

from sklearn.ensemble import GradientBoostingClassifier
from collections import Counter

@d6tflow.inherits(TaskTrainTestSplit)
class TaskTrainGradientBoostingClassifier(d6tflow.tasks.TaskPickle):
    n_estimators = luigi.IntParameter(default=100)
    learning_rate = luigi.FloatParameter(default=0.1)
    subsample = luigi.FloatParameter(default=1.0)

    def requires(self):
        return self.clone(TaskTrainTestSplit)

    def run(self):
        print(f"###Running {type(self).__name__}")

        X_train = self.input()["X_train"].load()
        y_train = self.input()["y_train"].load()
        X_test = self.input()["X_test"].load()
        y_test = self.input()["y_test"].load()

        train_counter = Counter(y_train)
        test_counter = Counter(y_test)
        print(f"Feature Distribution: Train: {train_counter[1] *100/ len(y_train)}%, Test: {test_counter[1] *100/ len(y_test)}%")

        model = GradientBoostingClassifier(n_estimators=self.n_estimators, learning_rate=self.learning_rate,  random_state=1, verbose=True)
        model.fit(X_train, y_train)
        self.save(model)


from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, accuracy_score
import numpy as np
from utils.plotter import confusion_matrix, evaluate_model, evaluate_predictions
from utils.plotter import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from utils.data_dumper import dump_json

@d6tflow.inherits(TaskTrainGradientBoostingClassifier, TaskTrainTestSplit)
class TaskEvaluateGradientBoostingClassifier(d6tflow.tasks.TaskPqPandas):

    def requires(self):
        return{"model": self.clone(TaskTrainGradientBoostingClassifier), "data": self.clone(TaskTrainTestSplit)}

    def run(self):
        print(f"###Running {type(self).__name__}")


        model = self.input()["model"].load()
        X_train = self.input()["data"]["X_train"].load()
        y_train = self.input()["data"]["y_train"].load()
        X_test = self.input()["data"]["X_test"].load()
        y_test = self.input()["data"]["y_test"].load()
        print(f"Length Train: {len(X_train)}, length Test {len(X_test)}")

        # Training predictions
        train_rf_predictions = model.predict(X_train)
        train_rf_probs = model.predict_proba(X_train)[:, 1]

        # Testing predictions (to determine performance)
        rf_predictions = model.predict(X_test)
        rf_probs = model.predict_proba(X_test)[:, 1]


        # Plot formatting
        #plt.style.use('fivethirtyeight')
        plt.rcParams['font.size'] = 18

        metrics = {}
        metrics.update(evaluate_predictions("test", rf_predictions, rf_probs, y_test))
        metrics.update(evaluate_predictions("train", train_rf_predictions, train_rf_probs, y_train))

        #metrics = evaluate_model(self.task_id, rf_predictions, rf_probs, y_test,  train_rf_predictions, train_rf_probs, y_train, train_dev_rf_predictions, train_dev_rf_probs, y_train_dev)


        # Confusion matrix
        cm = confusion_matrix(y_test, rf_predictions)
        cm_normalized = confusion_matrix(y_test, rf_predictions, normalize='all')

        #cm_values = plot_confusion_matrix(self.task_id, cm, classes = ['0', '1'],
        #                    title = 'Confusion Matrix', normalize=True)

        #Write to file
        results = {**metrics, "cm": cm, "cm_normalized": cm_normalized}
        dump_json(self.task_id, self.__dict__["param_kwargs"], results)


        # save test result
        evaluation_results = pd.DataFrame(zip(X_test, y_test, rf_predictions, rf_probs), columns=["x", "ground_truth", "predicted", "probability"])
        self.save(evaluation_results)