from preprocessing.helper import process_general_data
from tasks.preprocessing import TaskRuleProcessor, ProblemType
import d6tflow
import luigi
import os
from rules.return_none import ReturnNoneRule
from rules.condition_with_comparison import ConditionComparison
from rules.condition_comparison_simple import ConditionComparisonSimple
from tqdm.autonotebook import tqdm
import logging

@d6tflow.inherits(TaskRuleProcessor)
class TaskPrepareXYTransformer(d6tflow.tasks.TaskPickle):
    problem_type = luigi.EnumParameter(enum=ProblemType)
    window_size = luigi.IntParameter(default=20)
    step_size = luigi.IntParameter(default=3)

    def requires(self):
        return {"data": self.clone(TaskRuleProcessor)}

    def run(self):
        print(f"###Running {type(self).__name__}")

        data = self.input()["data"].load()

        x,y = process_general_data(data, None, window_size=self.window_size, step_size=self.step_size, problem_type=self.problem_type.value, encode_type=False, dont_encode=True)

        self.save((x,y))

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

@d6tflow.inherits(TaskPrepareXYTransformer)
class TaskTrainTestSplitTransformer(d6tflow.tasks.TaskPickle):
    test_split_percentage = luigi.FloatParameter(default=0.2)
    train_dev_split_percentage = luigi.FloatParameter(default=0.1)
    oversampling_enabled = luigi.BoolParameter(default=True)
    ratio_after_oversampling = luigi.FloatParameter(default=0.5)
    undersampling_enabled = luigi.BoolParameter(default=False)
    ratio_after_undersampling = luigi.FloatParameter(default=0.5)

    persist=['X_train','y_train', "X_train_dev", "y_train_dev", "X_test", "y_test"]

    def requires(self):
        return self.clone(TaskPrepareXYTransformer)

    def run(self):
        print(f"###Running {type(self).__name__}")

        #train/test split
        x,y = self.input().load()
        c1 = Counter()
        c1.update(y)
        X_train, X_test, y_train, y_test  = train_test_split(x, y, test_size=self.test_split_percentage, random_state=1)
        c2 = Counter()
        c2.update(y_train)
        print("Before", c1.most_common(), "After", c2.most_common())

        #split train into train and train_dev
        #calc the train_dev percentage of the train size
        train_dev_percent = self.train_dev_split_percentage / (1 - self.test_split_percentage)
        X_train, X_train_dev, y_train, y_train_dev  = train_test_split(X_train, y_train, test_size=train_dev_percent, random_state=1)
        
        print(f"BEFORE OVER/UNDERSAMPLING: Length Train: {len(X_train)}, length train_dev {len(X_train_dev)}, length Test {len(X_test)}")


        if self.oversampling_enabled:
            oversample = RandomOverSampler(sampling_strategy=self.ratio_after_oversampling, random_state=1)
            X_train, y_train = oversample.fit_resample(X_train, y_train)

        if self.undersampling_enabled:
            undersample = RandomUnderSampler(sampling_strategy=self.ratio_after_undersampling, random_state=1)
            X_train, y_train = undersample.fit_resample(X_train, y_train)

        print(f"AFTER OVER/UNDERSAMPLING: Length Train: {len(X_train)}, length train_dev {len(X_train_dev)}, length Test {len(X_test)}")
        self.save({
            "X_train": X_train,
            "y_train": y_train,
            "X_train_dev": X_train_dev,
            "y_train_dev": y_train_dev,
            "X_test": X_test,
            "y_test": y_test
        })


from simpletransformers.classification import ClassificationModel, ClassificationArgs
from collections import Counter
import pandas as pd

@d6tflow.inherits(TaskTrainTestSplitTransformer)
class TaskTrainTransformer(d6tflow.tasks.TaskPickle):
    n_trees_in_forest = luigi.IntParameter(default=100)
    max_features = luigi.Parameter(default="sqrt")
    model_type = luigi.Parameter(default="roberta")
    model_name = luigi.Parameter(default="huggingface/CodeBERTa-small-v1")

    def requires(self):
        return self.clone(TaskTrainTestSplitTransformer)

    def run(self):
        print(f"###Running {type(self).__name__}")

        X_train = self.input()["X_train"].load()
        y_train = self.input()["y_train"].load()
        X_test = self.input()["X_test"].load()
        y_test = self.input()["y_test"].load()

        train_counter = Counter(y_train)
        test_counter = Counter(y_test)
        print(f"Feature Distribution: Train: {train_counter[1] *100/ len(y_train)}%, Test: {test_counter[1] *100/ len(y_test)}%")

        #construct pandas dataframe
        zipped = zip(X_train, y_train)
        train_df = pd.DataFrame(zipped)
        train_df.columns = ["text", "labels"]

        model_args = ClassificationArgs(num_train_epochs=1)
        model = ClassificationModel(
            self.model_type, self.model_name, args=model_args, use_cuda=False
        )
        model.train_model(train_df)

        self.save(model)

from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, accuracy_score
import numpy as np
from utils.plotter import confusion_matrix, evaluate_model
from utils.plotter import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from utils.data_dumper import dump_json


@d6tflow.inherits(TaskTrainTransformer, TaskTrainTestSplitTransformer)
class TaskEvaluateTransformer(d6tflow.tasks.TaskPqPandas):

    def requires(self):
        return{"model": self.clone(TaskTrainTransformer), "data": self.clone(TaskTrainTestSplitTransformer)}

    def run(self):
        print(f"###Running {type(self).__name__}")


        model = self.input()["model"].load()
        X_train = self.input()["data"]["X_train"].load()
        y_train = self.input()["data"]["y_train"].load()
        X_train_dev = self.input()["data"]["X_train_dev"].load()
        y_train_dev = self.input()["data"]["y_train_dev"].load()
        X_test = self.input()["data"]["X_test"].load()
        y_test = self.input()["data"]["y_test"].load()
        print(f"Length Train: {len(X_train)}, length Test {len(X_test)}")

        # Training predictions
        train_rf_predictions, _ = model.predict(X_train)
        #train_rf_probs = model.predict_proba(X_train)[:, 1]
        train_rf_probs = np.zeros(len(train_rf_predictions))

        #Train dev predictions
        train_dev_rf_predictions, _= model.predict(X_train_dev)
        #train_dev_rf_probs = model.predict_proba(X_train_dev)[:, 1]
        train_dev_rf_probs = np.zeros(len(train_dev_rf_predictions))

        # Testing predictions (to determine performance)
        rf_predictions, _ = model.predict(X_test)
        #rf_probs = model.predict_proba(X_test)[:, 1]
        rf_probs = np.zeros(len(rf_predictions))

        # Plot formatting
        #plt.style.use('fivethirtyeight')
        plt.rcParams['font.size'] = 18


        metrics = evaluate_model(self.task_id, rf_predictions, rf_probs, y_test,  train_rf_predictions, train_rf_probs, y_train, train_dev_rf_predictions, train_dev_rf_probs, y_train_dev)


        # Confusion matrix
        cm = confusion_matrix(y_test, rf_predictions)
        cm_values = plot_confusion_matrix(self.task_id, cm, classes = ['0', '1'],
                            title = 'Confusion Matrix', normalize=True)

        #Write to file
        results = {**metrics, **cm_values}
        dump_json(self.task_id, self.__dict__["param_kwargs"], results)


        # save test result
        evaluation_results = pd.DataFrame(zip(X_test, y_test, rf_predictions, rf_probs), columns=["x", "ground_truth", "predicted", "probability"])
        self.save(evaluation_results)