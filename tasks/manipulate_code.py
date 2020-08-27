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
                    line = lines[i]
                    new_line = re.sub("return None", callback, line)
                    lines[i] = new_line
                code_string = "\n".join(lines)
                data[index]["src"] = code_string
        
        else:
            raise NotImplementedError("Only return none implemented right now")


        
        self.save(data)

