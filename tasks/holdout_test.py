import d6tflow
import luigi
import os
from tasks.preprocessing import ProblemType
from tasks.preprocessing import TaskRuleProcessor
from tqdm.autonotebook import tqdm
import logging
import random
import re
import pandas as pd

from models.random_forst import process_general_data
from tasks.preprocessing import TaskVocabCreator
from imblearn.under_sampling import RandomUnderSampler

class TaskPrepareXYHoldout(d6tflow.tasks.TaskPickle):
    window_size = luigi.IntParameter(default=20)
    step_size = luigi.IntParameter(default=3)
    encode_type = luigi.BoolParameter(default=True)
    vocab_input_directory = luigi.Parameter(default="final_dataset")
    test_input_directory = luigi.Parameter(default="final_validation")
    max_vocab_size = luigi.IntParameter(default=100000)
    problem_type = luigi.EnumParameter(enum=ProblemType)

    def requires(self):
        return {"data": TaskRuleProcessor(input_src_path=self.test_input_directory), "vocab": TaskVocabCreator(max_vocab_size=self.max_vocab_size, input_src_path=self.vocab_input_directory)}

    def run(self):
        print(f"###Running {type(self).__name__}")


        data = self.input()["data"].load()
        vocab = self.input()["vocab"].load()

        # prepare XY
        x,y = process_general_data(data, vocab, window_size=self.window_size, step_size=self.step_size, problem_type=self.problem_type.value, encode_type=self.encode_type)

        print(f"Length dataset: {len(x)}")

        self.save((x,y))

from utils.data_dumper import dump_json
from utils.plotter import confusion_matrix, evaluate_model, plot_confusion_matrix, evaluate_predictions
from sklearn.metrics import confusion_matrix

@d6tflow.inherits(TaskPrepareXYHoldout)
class TaskEvalHoldoutEnsemble(d6tflow.tasks.TaskPickle):
    model  = luigi.Parameter()
    training_parameter = luigi.Parameter()


    def requires(self):
        return self.clone(TaskPrepareXYHoldout)

    def run(self):
        print(f"###Running {type(self).__name__}")

        x,y = self.input().load()

        #predict
        rf_predictions = self.model.predict(x)
        rf_probs = self.model.predict_proba(x)[:, 1]

        #evaluate
        metrics = {}
        metrics.update(evaluate_predictions("holdout", rf_predictions, rf_probs, y))

        #confusion matrix
        cm = confusion_matrix(y, rf_predictions)
        cm_normalized = confusion_matrix(y, rf_predictions, normalize='all')


        #Write to file
        results = {**metrics,  "holdout_cm": cm, "holdout_cm_normalized": cm_normalized}
        parameter_dict = {k:v for (k,v) in self.__dict__["param_kwargs"].items() if k != "model"}
        dump_json(self.task_id, parameter_dict, results)

        evaluation_results = pd.DataFrame(zip(x, y, rf_predictions, rf_probs), columns=["x", "ground_truth", "predicted", "probability"])
        self.save(evaluation_results)

from utils.data_dumper import dump_json
from utils.plotter import confusion_matrix, evaluate_model, plot_confusion_matrix, evaluate_predictions
from sklearn.metrics import confusion_matrix

@d6tflow.inherits(TaskPrepareXYHoldout)
class TaskEvalHoldoutKeras(d6tflow.tasks.TaskPickle):
    model  = luigi.Parameter()
    training_parameter = luigi.Parameter()

    def requires(self):
        return self.clone(TaskPrepareXYHoldout)

    def run(self):
        print(f"###Running {type(self).__name__}")

        x,y = self.input().load()


        #predict
        pred_raw = self.model.predict(x, verbose=1)
        rf_predictions = (pred_raw > 0.5).astype("int32")
        rf_probs = pred_raw

        #evaluate
        metrics = {}
        metrics.update(evaluate_predictions("holdout", rf_predictions, rf_probs, y))

        #confusion matrix
        cm = confusion_matrix(y, rf_predictions)
        cm_normalized = confusion_matrix(y, rf_predictions, normalize='all')


        #Write to file
        results = {**metrics,  "holdout_cm": cm, "holdout_cm_normalized": cm_normalized}
        parameter_dict = {k:v for (k,v) in self.__dict__["param_kwargs"].items() if k != "model"}
        dump_json(self.task_id, parameter_dict, results)

        evaluation_results = pd.DataFrame(zip(x, y, rf_predictions, rf_probs), columns=["x", "ground_truth", "predicted", "probability"])
        self.save(evaluation_results)
