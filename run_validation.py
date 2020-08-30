from tasks.preprocessing import ProblemType, TaskVocabCreator
from tasks.gradient_boosting_classifier import TaskTrainGradientBoostingClassifier
from tasks.svm import TaskTrainSVM
from tasks.lstm import TaskTrainLstm
from tasks.random_forest import TaskTrainRandomForest
from tasks.manipulate_code import TaskEvalEnsemble, TaskEvalKeras
import d6tflow

d6tflow.settings.log_level = 'DEBUG' # 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'

model = TaskTrainLstm(max_vocab_size=100000, input_src_path="second_large_dataset", problem_type=ProblemType.RETURN_NONE, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.5, embedding_vecor_length=32, epochs=3, batch_size=64, encode_type=False, num_lstm_cells=10, dropout_emb_lstm=0.2, dropout_lstm_dense=0.2).outputLoad()



t = TaskEvalKeras(model=model, input_src_path="validation", problem_type=ProblemType.RETURN_NONE, vocab_input_directory="second_large_dataset", max_vocab_size=100000, encode_type=False)
d6tflow.run(t)