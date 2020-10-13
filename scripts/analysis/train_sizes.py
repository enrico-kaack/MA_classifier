from tasks.preprocessing import TaskTrainTestSplit, ProblemType, TaskPrepareXY
import d6tflow
from collections import Counter

d6tflow.settings.log_level = 'WARNING' # 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'

def print_train_size(oversampling, undersampling):
    for problem_type in ProblemType:
        t.TaskTrainTestSplit(problem_type=problem_type, oversampling_enabled=oversampling, ratio_after_oversampling=0.5, undersampling_enabled=undersampling, ratio_after_undersampling=0.5,encode_type=True)
        y = t.output()["y_train"].load()
        c = Counter()
        c.update(y)
        lablel_0_value, label_0 = c.most_common()[0]
        label_1_value, label_1 = c.most_common()[1]

        print("\n",problem_type, f"Oversampling: {t.oversampling_enabled} {t.ratio_after_oversampling}|Undersampling: {t.undersampling_enabled} {t.ratio_after_undersampling}")
        print(f"Label {lablel_0_value}: {label_0}")
        print(f"Label {label_1_value}: {label_1}")
        print(f"Total length: {len(y)}")



print("Not invalidating datasets, please delete data/TaskPrepareXY and data/TaskRuleProcessor and data/TaskSourceFileToDataStructure")
print("rm -r data/TaskPrepareXY data/TaskRuleProcessor data/TaskSourceFileToDataStructure")


print("Final dataset TRAIN")

print_train_size(TaskTrainTestSplit(False, False)
print_train_size(TaskTrainTestSplit(True, False)
print_train_size(TaskTrainTestSplit(False, True)

