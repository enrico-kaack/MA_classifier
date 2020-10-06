from tasks.preprocessing import TaskTrainTestSplit, ProblemType, TaskPrepareXY
import d6tflow
from collections import Counter

d6tflow.settings.log_level = 'WARNING' # 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'

def print_dataset_class_distribution(problem_type, src_path):
    t = TaskPrepareXY(input_src_path=src_path, problem_type=problem_type)
    d6tflow.run(t)
    _, y = t.outputLoad()
    c = Counter()
    c.update(y)
    label_0 = c.most_common()[0][1]
    label_1 = c.most_common()[1][1]

    print(problem_type)
    print(f"Label 0: {label_0 / len(y) * 100}%")
    print(f"Label 1: {label_1 / len(y) * 100}%\n")

print("Not invalidating datasets, please delete data/TaskPrepareXY and data/TaskRuleProcessor and data/TaskSourceFileToDataStructure")
print("rm -r data/TaskPrepareXY data/TaskRuleProcessor data/TaskSourceFileToDataStructure")


print("Final dataset TRAIN")
print_dataset_class_distribution(ProblemType.RETURN_NONE, "final_dataset")
print_dataset_class_distribution(ProblemType.CONDITION_COMPARISON_SIMPLE, "final_dataset")
print_dataset_class_distribution(ProblemType.CONDITION_COMPARISON, "final_dataset")

print("Final test TEST")
print_dataset_class_distribution(ProblemType.RETURN_NONE, "final_test")
print_dataset_class_distribution(ProblemType.CONDITION_COMPARISON_SIMPLE, "final_test")
print_dataset_class_distribution(ProblemType.CONDITION_COMPARISON, "final_test")

print("Final holdout HOLDOUT")
print_dataset_class_distribution(ProblemType.RETURN_NONE, "final_validation")
print_dataset_class_distribution(ProblemType.CONDITION_COMPARISON_SIMPLE, "final_validation")
print_dataset_class_distribution(ProblemType.CONDITION_COMPARISON, "final_validation")