from tasks.preprocessing import TaskRuleProcessor, ProblemType, TaskSourceFileToDataStructure
from input.input_data_analysis import analyse_parsed_data, analyse_problems_per_project, analyse_files_per_project, analyse_total_file_size
import d6tflow
d6tflow.settings.log_level = 'WARNING' # 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'

# analyse the different datasets (train, test, validation) with different metrics.
print("final_dataset TRAIN")
d6tflow.run(TaskRuleProcessor(input_src_path="final_dataset"))
problems = TaskRuleProcessor(input_src_path="final_dataset").outputLoad()
analyse_parsed_data(problems)
print("\nTotal File Size")
analyse_total_file_size(problems)
print("\nProblems per project")
analyse_problems_per_project(problems, 2)
print("\nFiles per project")
analyse_files_per_project(problems, 2)

print("final_test TEST")
d6tflow.run(TaskRuleProcessor(input_src_path="final_test"))
problems = TaskRuleProcessor(input_src_path="final_test").outputLoad()
analyse_parsed_data(problems)
print("\nTotal File Size")
analyse_total_file_size(problems)
print("\nProblems per project")
analyse_problems_per_project(problems, 3)
print("\nFiles per project")
analyse_files_per_project(problems, 2)

print("final_valdiation HOLDOUT")
d6tflow.run(TaskRuleProcessor(input_src_path="final_validation"))
problems = TaskRuleProcessor(input_src_path="final_validation").outputLoad()
analyse_parsed_data(problems)
print("\nTotal File Size")
analyse_total_file_size(problems)
print("\nProblems per project")
analyse_problems_per_project(problems, 4)
print("\nFiles per project")
analyse_files_per_project(problems, 4)