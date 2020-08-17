import d6tflow
import luigi
import os
from rules.return_none import ReturnNoneRule
from tqdm.autonotebook import tqdm
import logging

class TaskSourceFileToDataStructure(d6tflow.tasks.TaskPickle):
    input_src_path = luigi.Parameter(default="raw_data")

    def run(self):
        print(f"###Running {type(self).__name__}")

        assert(os.path.isdir(self.input_src_path), "Input_src_path needs to be a directory")
        data = []
        for dir_path, dir_names, file_names in os.walk(self.input_src_path):
            for f in file_names:
                _, file_extension = os.path.splitext(f)
                if file_extension == ".py":
                    file_path = os.path.join(dir_path, f)
                    with open(file_path, "r") as open_file:
                        file_content = open_file.read()
                        data.append({
                            "file_path": file_path,
                            "src": file_content
                        })
        self.save(data)

@d6tflow.inherits(TaskSourceFileToDataStructure)
class TaskRuleProcessor(d6tflow.tasks.TaskPickle):

    def requires(self):
        return self.clone(TaskSourceFileToDataStructure)

    def run(self):
        print(f"###Running {type(self).__name__}")

        dataset = self.input().load()
        rules = [ReturnNoneRule()]
        processed_data = []
        for data in tqdm(dataset):
            problems = []
            for rule in rules:
                try:
                    problems.extend( rule.analyse_source_code(data["src"]))
                except SyntaxError:
                    logging.warn(f"Syntax error on file:{data['file_path']}. Ignoring file")
            processed_data.append({**data, "problems": problems})
        self.save(processed_data)

from io import BytesIO
from tokenize import tokenize
from collections import Counter
from functools import reduce

@d6tflow.inherits(TaskSourceFileToDataStructure)
class TaskVocabCreator(d6tflow.tasks.TaskPickle):
    max_vocab_size = luigi.Parameter(default=1000)

    def requires(self):
        return self.clone(TaskSourceFileToDataStructure)

    def run(self):
        print(f"###Running {type(self).__name__}")

        data = self.input().load()

        #use the counter to count token values
        c = Counter()
        for d in tqdm(data):
            src_code = d['src']
            tokens= tokenize(BytesIO(src_code.encode('utf-8')).readline)
            for _, tokval, _, _, _ in tokens:
                c.update([tokval.lower()])

        vocab_dict= {}
        for i, vocab in enumerate(c.most_common(self.max_vocab_size)):
            vocab_dict[vocab[0]] = i


        #analyse unknown token occurence
        all_tokens = c.most_common()
        common_tokens = all_tokens[:self.max_vocab_size]
        non_common_tokens = all_tokens[self.max_vocab_size+1:]
        count_common_tokens = reduce(lambda x,y: x[1]+y[1], common_tokens)
        count_uncommon_tokens = reduce(lambda x,y: x[1]+y[1], non_common_tokens)
        print(f"Unknown Token Frequency: {count_uncommon_tokens*100/(count_common_tokens+count_uncommon_tokens)}%")

        self.save(vocab_dict)

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
        print(f"###Running {type(self).__name__}")

        data, vocab = self.inputLoad()

        x,y = process_general_data(data, vocab, window_size=self.window_size, step_size=self.step_size, problem_type=self.problem_type.value)

        self.save((x,y))

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

@d6tflow.inherits(TaskPrepareXY)
class TaskTrainTestSplit(d6tflow.tasks.TaskPickle):
    test_split_percentage = luigi.FloatParameter(default=0.25)
    oversampling_enabled = luigi.BoolParameter(default=True)
    ratio_after_oversampling = luigi.FloatParameter(default=0.5)
    undersampling_enabled = luigi.BoolParameter(default=False)
    ratio_after_undersampling = luigi.FloatParameter(default=0.5)

    persist=['X_train','y_train', "X_test", "y_test"]

    def requires(self):
        return self.clone(TaskPrepareXY)

    def run(self):
        print(f"###Running {type(self).__name__}")

        x,y = self.input().load()

        X_train, X_test, y_train, y_test  = train_test_split(x, y, test_size=self.test_split_percentage, random_state=1)

        if self.oversampling_enabled:
            oversample = RandomOverSampler(sampling_strategy=self.ratio_after_oversampling, random_state=1)
            X_train, y_train = oversample.fit_resample(X_train, y_train)

        if self.undersampling_enabled:
            undersample = RandomUnderSampler(sampling_strategy=self.ratio_after_undersampling, random_state=1)
            X_train, y_train = undersample.fit_resample(X_train, y_train)

        print(f"Length Train: {len(X_train)}, length Test {len(X_test)}")
        self.save({
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test
        })