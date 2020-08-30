from tasks.preprocessing import ProblemType, TaskVocabCreator
from tasks.gradient_boosting_classifier import TaskTrainGradientBoostingClassifier
from tasks.svm import TaskTrainSVM
from tasks.lstm import TaskTrainLstm
from tasks.random_forest import TaskTrainRandomForest
from tasks.manipulate_code import TaskEvalEnsemble
import d6tflow

d6tflow.settings.log_level = 'WARNING' # 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'

model = TaskTrainRandomForest(max_vocab_size=100000, input_src_path="second_large_dataset", problem_type=ProblemType.RETURN_NONE, oversampling_enabled=True, ratio_after_oversampling=0.5).outputLoad()



t = TaskEvalEnsemble(model=model, input_src_path="second_large_dataset", problem_type=ProblemType.RETURN_NONE, vocab_input_directory="second_large_dataset", max_vocab_size=100000)
d6tflow.run(t)