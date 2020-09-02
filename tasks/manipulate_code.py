import d6tflow
import luigi
import os
from tasks.preprocessing import ProblemType
from tasks.preprocessing import TaskRuleProcessor
from tqdm.autonotebook import tqdm
import logging
import random
import re


@d6tflow.inherits(TaskRuleProcessor)
class TaskCodeManipulator(d6tflow.tasks.TaskPickle):
    problem_type = luigi.EnumParameter(enum=ProblemType)

    def requires(self):
        return self.clone(TaskRuleProcessor)

    def run(self):
        print(f"###Running {type(self).__name__}")

        data = self.input().load()

        if self.problem_type == ProblemType.RETURN_NONE:
            replacements = ["return a if a < 2 else None", "return None if a < 2 else a"]
            def callback(matchobj):
                return random.choice(replacements)
            for index, d in enumerate(data):
                problem_line_numbers = [l['line_number'] for l in d['problems'] if l['type'] == self.problem_type]

                code_string = d["src"]
                lines = code_string.splitlines(keepends=False)
                for i in problem_line_numbers:
                    line = lines[i-1] #-1 since line numbers are indexed from 1, the array from 0
                    new_line = re.sub("return None", callback, line)
                    lines[i-1] = new_line
                code_string = "\n".join(lines)
                data[index]["src"] = code_string

        if self.problem_type == ProblemType.CONDITION_COMPARISON: #trained on normal, validated on simple, therefore normal condition here 
            replacements = ["if a<b or b<a:", "if is_smaller(a,b) and b < c:", "if not a < b:"]
            def callback(matchobj):
                return random.choice(replacements)
            for index, d in enumerate(data):
                problem_line_numbers = [l['line_number'] for l in d['problems'] if l['type'] == self.problem_type]

                code_string = d["src"]
                lines = code_string.splitlines(keepends=False)
                for i in problem_line_numbers:
                    line = lines[i-1] #-1 since line numbers are indexed from 1, the array from 0
                    new_line = re.sub("if .*:", callback, line)
                    lines[i-1] = new_line
                code_string = "\n".join(lines)
                data[index]["src"] = code_string
        
        else:
            raise NotImplementedError("Only return none implemented right now")

        self.save(data)


from models.random_forst import process_general_data
from tasks.preprocessing import TaskVocabCreator

@d6tflow.inherits(TaskCodeManipulator)
class TaskPrepareXYValidation(d6tflow.tasks.TaskPickle):
    window_size = luigi.IntParameter(default=20)
    step_size = luigi.IntParameter(default=3)
    encode_type = luigi.BoolParameter(default=True)
    vocab_input_directory = luigi.Parameter(default="second_large_dataset")
    max_vocab_size = luigi.IntParameter(default=100000)

    def requires(self):
        return {"data": self.clone(TaskCodeManipulator), "vocab": TaskVocabCreator(max_vocab_size=self.max_vocab_size, input_src_path=self.vocab_input_directory)}

    def run(self):
        print(f"###Running {type(self).__name__}")


        data = self.input()["data"].load()
        vocab = self.input()["vocab"].load()

        # prepare XY
        x,y = process_general_data(data, vocab, window_size=self.window_size, step_size=self.step_size, problem_type=self.problem_type.value, encode_type=self.encode_type)
        print(f"Length dataset: {len(x)}")

        self.save((x,y))

from utils.data_dumper import dump_json
from utils.plotter import confusion_matrix, evaluate_model, plot_confusion_matrix
from sklearn.metrics import confusion_matrix

@d6tflow.inherits(TaskPrepareXYValidation)
class TaskEvalEnsemble(d6tflow.tasks.TaskPickle):
    model  = luigi.Parameter()

    def requires(self):
        return self.clone(TaskPrepareXYValidation)

    def run(self):
        print(f"###Running {type(self).__name__}")

        x,y = self.input().load()

        #predict
        rf_predictions = self.model.predict(x)
        rf_probs = self.model.predict_proba(x)[:, 1]

        #evaluate
        metrics = evaluate_model(self.task_id, rf_predictions, rf_probs, y,  [], [], [], only_test=True)


        # Confusion matrix
        cm = confusion_matrix(y, rf_predictions)
        cm_values = plot_confusion_matrix(self.task_id, cm, classes = ['0', '1'],
                            title = 'Confusion Matrix', normalize=True)

        #Write to file
        results = {**metrics, **cm_values}
        parameter_dict = {k:v for (k,v) in self.__dict__["param_kwargs"].items() if k != "model"}
        dump_json(self.task_id, parameter_dict, results)

from utils.data_dumper import dump_json
from utils.plotter import confusion_matrix, evaluate_model, plot_confusion_matrix
from sklearn.metrics import confusion_matrix

@d6tflow.inherits(TaskPrepareXYValidation)
class TaskEvalKeras(d6tflow.tasks.TaskPickle):
    model  = luigi.Parameter()

    def requires(self):
        return self.clone(TaskPrepareXYValidation)

    def run(self):
        print(f"###Running {type(self).__name__}")

        x,y = self.input().load()

        #predict
        rf_predictions = (self.model.predict(x) > 0.5).astype("int32")
        rf_probs = self.model.predict(x)

        #evaluate
        metrics = evaluate_model(self.task_id, rf_predictions, rf_probs, y,  [], [], [], only_test=True)


        # Confusion matrix
        cm = confusion_matrix(y, rf_predictions)
        cm_values = plot_confusion_matrix(self.task_id, cm, classes = ['0', '1'],
                            title = 'Confusion Matrix', normalize=True)

        #Write to file
        results = {**metrics, **cm_values}
        parameter_dict = {k:v for (k,v) in self.__dict__["param_kwargs"].items() if k != "model"}
        dump_json(self.task_id, parameter_dict, results)
