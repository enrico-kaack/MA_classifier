from tasks.preprocessing import ProblemType, TaskVocabCreator
from tasks.gradient_boosting_classifier import TaskTrainGradientBoostingClassifier, TaskEvaluateGradientBoostingClassifier
from tasks.svm import TaskTrainSVM, TaskEvaluateSVM
from tasks.lstm import TaskTrainLstm, TaskEvaluateLstm
from tasks.random_forest import TaskTrainRandomForest, TaskEvaluateRandomForest
from tasks.manipulate_code import TaskEvalEnsemble, TaskEvalKeras, TaskPrepareXYValidation
import d6tflow

source = "second_large_dataset"
for problem_type in ProblemType:
    t =[ 
        TaskEvaluateGradientBoostingClassifier(max_vocab_size=100000, input_src_path=source, problem_type=problem_type, oversampling_enabled=True, undersampling_enabled=False, ratio_after_oversampling=0.5, learning_rate=0.2, n_estimators=200, subsample=1.0),
        TaskEvaluateGradientBoostingClassifier(max_vocab_size=100000, input_src_path=source, problem_type=problem_type, oversampling_enabled=True, undersampling_enabled=False, ratio_after_oversampling=0.5, learning_rate=0.2, n_estimators=200, subsample=0.4),
        TaskEvaluateGradientBoostingClassifier(max_vocab_size=100000, input_src_path=source, problem_type=problem_type, oversampling_enabled=True, undersampling_enabled=False, ratio_after_oversampling=0.5, learning_rate=0.2, n_estimators=200, subsample=0.7),
        TaskEvaluateRandomForest(max_vocab_size=100000, input_src_path=source, problem_type=problem_type, oversampling_enabled=True, ratio_after_oversampling=0.5, encode_type=True, class_weight="balanced"),
        TaskEvaluateRandomForest(max_vocab_size=100000, input_src_path=source, problem_type=problem_type, oversampling_enabled=False, undersampling_enabled=True,  ratio_after_undersampling=0.5, encode_type=True, class_weight="balanced"),
        TaskEvaluateLstm(max_vocab_size=100000, input_src_path=source, problem_type=problem_type, oversampling_enabled=False, undersampling_enabled=False,  embedding_vecor_length=32, epochs=3, batch_size=256, encode_type=False, num_lstm_cells=10, dropout_emb_lstm=0.2, dropout_lstm_dense=0.2),
        TaskEvaluateLstm(max_vocab_size=100000, input_src_path=source, problem_type=problem_type, oversampling_enabled=False, undersampling_enabled=False,  embedding_vecor_length=32, epochs=3, batch_size=512, encode_type=False, num_lstm_cells=10, dropout_emb_lstm=0.2, dropout_lstm_dense=0.2),
    ]

    d6tflow.run(t)
