import d6tflow
import luigi
import os
from rules.return_none import ReturnNoneRule
from tqdm.autonotebook import tqdm
import logging

class TaskSourceFileToDataStructure(d6tflow.tasks.TaskPickle):
    input_src_path = luigi.Parameter(default="raw_data")

    def run(self):
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


@d6tflow.inherits(TaskSourceFileToDataStructure)
class TaskVocabCreator(d6tflow.tasks.TaskPickle):
    max_vocab_size = luigi.Parameter(default=1000)

    def requires(self):
        return self.clone(TaskSourceFileToDataStructure)

    def run(self):
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
        self.save(vocab_dict)

