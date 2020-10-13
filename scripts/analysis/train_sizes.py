from tasks.preprocessing import TaskTrainTestSplit, ProblemType, TaskPrepareXY
import d6tflow
from collections import Counter

d6tflow.settings.log_level = 'WARNING' # 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'

def print_train_size(t):
    for problem_type in ProblemType:
        t.problem_type = problem_type
        y = t.outputLoad("y_train")
        c = Counter()
        c.update(y)
        lablel_0_value, label_0 = c.most_common()[0]
        label_1_value, label_1 = c.most_common()[1]

        print("\n",problem_type, f"Oversampling: {t.oversampling_enabled} {t.ratio_after_oversampling}|Undersampling: {t.undersampling_enabled} {t.ratio_after_undersampling}")
        print(f"Label {lablel_0_value}: {label_0}")
        print(f"Label {label_1_value}: {label_1}\n")
        print(f"Total length: {len(y)}")



print("Not invalidating datasets, please delete data/TaskPrepareXY and data/TaskRuleProcessor and data/TaskSourceFileToDataStructure")
print("rm -r data/TaskPrepareXY data/TaskRuleProcessor data/TaskSourceFileToDataStructure")


print("Final dataset TRAIN")

print_train_size(TaskTrainTestSplit(problem_type=ProblemType.RETURN_NONE, oversampling_enabled=False, ratio_after_oversampling=0.5, undersampling_enabled=False, ratio_after_undersampling=0.5,encode_type=True))
print_train_size(TaskTrainTestSplit(problem_type=ProblemType.RETURN_NONE, oversampling_enabled=True, ratio_after_oversampling=0.5, undersampling_enabled=False, ratio_after_undersampling=0.5,encode_type=True))
print_train_size(TaskTrainTestSplit(problem_type=ProblemType.RETURN_NONE, oversampling_enabled=False, ratio_after_oversampling=0.5, undersampling_enabled=True, ratio_after_undersampling=0.5,encode_type=True))

