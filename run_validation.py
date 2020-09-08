from tasks.preprocessing import ProblemType, TaskVocabCreator
from tasks.gradient_boosting_classifier import TaskTrainGradientBoostingClassifier
from tasks.svm import TaskTrainSVM
from tasks.lstm import TaskTrainLstm
from tasks.random_forest import TaskTrainRandomForest
from tasks.manipulate_code import TaskEvalEnsemble, TaskEvalKeras, TaskPrepareXYValidation
import d6tflow

d6tflow.settings.log_level = 'INFO' # 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'

train_source = "second_large_dataset"
validation_source = "validation"
models_ensemble = [
        TaskTrainRandomForest(max_vocab_size=100000, input_src_path=train_source, problem_type=ProblemType.RETURN_NONE, oversampling_enabled=True, ratio_after_oversampling=0.5),
        TaskTrainRandomForest(max_vocab_size=100000, input_src_path=train_source, problem_type=ProblemType.RETURN_NONE, oversampling_enabled=True, ratio_after_oversampling=1.0),
        TaskTrainRandomForest(max_vocab_size=100000, input_src_path=train_source, problem_type=ProblemType.RETURN_NONE, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.5),
        TaskTrainRandomForest(max_vocab_size=100000, input_src_path=train_source, problem_type=ProblemType.RETURN_NONE, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.1),
        TaskTrainGradientBoostingClassifier(max_vocab_size=100000, input_src_path=train_source, problem_type=ProblemType.RETURN_NONE, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.5, learning_rate=0.1, n_estimators=100),
        TaskTrainGradientBoostingClassifier(max_vocab_size=100000, input_src_path=train_source, problem_type=ProblemType.RETURN_NONE, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.5, learning_rate=0.2, n_estimators=100),
        TaskTrainGradientBoostingClassifier(max_vocab_size=100000, input_src_path=train_source, problem_type=ProblemType.RETURN_NONE, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.5, learning_rate=0.01, n_estimators=1000),
        TaskTrainRandomForest(max_vocab_size=100000, input_src_path=train_source, problem_type=ProblemType.RETURN_NONE, oversampling_enabled=True, ratio_after_oversampling=0.5, encode_type=False),
        TaskTrainRandomForest(max_vocab_size=100000, input_src_path=train_source, problem_type=ProblemType.RETURN_NONE, oversampling_enabled=True, ratio_after_oversampling=1.0, encode_type=False),
        TaskTrainRandomForest(max_vocab_size=100000, input_src_path=train_source, problem_type=ProblemType.RETURN_NONE, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.5, encode_type=False),

        TaskTrainGradientBoostingClassifier(max_vocab_size=100000, input_src_path=train_source, problem_type=ProblemType.RETURN_NONE, oversampling_enabled=True, ratio_after_oversampling=0.5, n_estimators = 200, learning_rate=0.2),
        TaskTrainGradientBoostingClassifier(max_vocab_size=100000, input_src_path=train_source, problem_type=ProblemType.RETURN_NONE, oversampling_enabled=True, ratio_after_oversampling=0.5, n_estimators = 300, learning_rate=0.2),
        TaskTrainGradientBoostingClassifier(max_vocab_size=100000, input_src_path=train_source, problem_type=ProblemType.RETURN_NONE, oversampling_enabled=True, ratio_after_oversampling=0.5, n_estimators = 400, learning_rate=0.2),
        TaskTrainGradientBoostingClassifier(max_vocab_size=100000, input_src_path=train_source, problem_type=ProblemType.RETURN_NONE, oversampling_enabled=True, ratio_after_oversampling=0.5, n_estimators = 200, learning_rate=0.1),
        TaskTrainGradientBoostingClassifier(max_vocab_size=100000, input_src_path=train_source, problem_type=ProblemType.RETURN_NONE, oversampling_enabled=True, ratio_after_oversampling=0.5, n_estimators = 300, learning_rate=0.1),
        TaskTrainGradientBoostingClassifier(max_vocab_size=100000, input_src_path=train_source, problem_type=ProblemType.RETURN_NONE, oversampling_enabled=True, ratio_after_oversampling=0.5, n_estimators = 400, learning_rate=0.1),



        TaskTrainRandomForest(max_vocab_size=100000, input_src_path=train_source, problem_type=ProblemType.CONDITION_COMPARISON_SIMPLE, oversampling_enabled=True, ratio_after_oversampling=0.5),
        TaskTrainRandomForest(max_vocab_size=100000, input_src_path=train_source, problem_type=ProblemType.CONDITION_COMPARISON_SIMPLE, oversampling_enabled=True, ratio_after_oversampling=1.0),
        TaskTrainRandomForest(max_vocab_size=100000, input_src_path=train_source, problem_type=ProblemType.CONDITION_COMPARISON_SIMPLE, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.5),
        TaskTrainRandomForest(max_vocab_size=100000, input_src_path=train_source, problem_type=ProblemType.CONDITION_COMPARISON_SIMPLE, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.1),
        TaskTrainGradientBoostingClassifier(max_vocab_size=100000, input_src_path=train_source, problem_type=ProblemType.CONDITION_COMPARISON_SIMPLE, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.5, learning_rate=0.1, n_estimators=100),
        TaskTrainGradientBoostingClassifier(max_vocab_size=100000, input_src_path=train_source, problem_type=ProblemType.CONDITION_COMPARISON_SIMPLE, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.5, learning_rate=0.2, n_estimators=100),
        TaskTrainGradientBoostingClassifier(max_vocab_size=100000, input_src_path=train_source, problem_type=ProblemType.CONDITION_COMPARISON_SIMPLE, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.5, learning_rate=0.01, n_estimators=1000),
        TaskTrainRandomForest(max_vocab_size=100000, input_src_path=train_source, problem_type=ProblemType.CONDITION_COMPARISON_SIMPLE, oversampling_enabled=True, ratio_after_oversampling=0.5, encode_type=False),
        TaskTrainRandomForest(max_vocab_size=100000, input_src_path=train_source, problem_type=ProblemType.CONDITION_COMPARISON_SIMPLE, oversampling_enabled=True, ratio_after_oversampling=1.0, encode_type=False),
        TaskTrainRandomForest(max_vocab_size=100000, input_src_path=train_source, problem_type=ProblemType.CONDITION_COMPARISON_SIMPLE, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.5, encode_type=False),

        TaskTrainGradientBoostingClassifier(max_vocab_size=100000, input_src_path=train_source, problem_type=ProblemType.CONDITION_COMPARISON_SIMPLE, oversampling_enabled=True, ratio_after_oversampling=0.5, n_estimators = 200, learning_rate=0.2),
        TaskTrainGradientBoostingClassifier(max_vocab_size=100000, input_src_path=train_source, problem_type=ProblemType.CONDITION_COMPARISON_SIMPLE, oversampling_enabled=True, ratio_after_oversampling=0.5, n_estimators = 300, learning_rate=0.2),
        TaskTrainGradientBoostingClassifier(max_vocab_size=100000, input_src_path=train_source, problem_type=ProblemType.CONDITION_COMPARISON_SIMPLE, oversampling_enabled=True, ratio_after_oversampling=0.5, n_estimators = 400, learning_rate=0.2),
        TaskTrainGradientBoostingClassifier(max_vocab_size=100000, input_src_path=train_source, problem_type=ProblemType.CONDITION_COMPARISON_SIMPLE, oversampling_enabled=True, ratio_after_oversampling=0.5, n_estimators = 200, learning_rate=0.1),
        TaskTrainGradientBoostingClassifier(max_vocab_size=100000, input_src_path=train_source, problem_type=ProblemType.CONDITION_COMPARISON_SIMPLE, oversampling_enabled=True, ratio_after_oversampling=0.5, n_estimators = 300, learning_rate=0.1),
        TaskTrainGradientBoostingClassifier(max_vocab_size=100000, input_src_path=train_source, problem_type=ProblemType.CONDITION_COMPARISON_SIMPLE, oversampling_enabled=True, ratio_after_oversampling=0.5, n_estimators = 400, learning_rate=0.1),


        TaskTrainSVM(max_vocab_size=100000, input_src_path=train_source, problem_type=ProblemType.RETURN_NONE, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.5, svm_kernel="rbf", svm_predict_probability=True),
        TaskTrainSVM(max_vocab_size=100000, input_src_path=train_source, problem_type=ProblemType.RETURN_NONE, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.1, svm_kernel="rbf", svm_predict_probability=True),
        TaskTrainSVM(max_vocab_size=100000, input_src_path=train_source, problem_type=ProblemType.RETURN_NONE, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.5, svm_kernel="rbf", svm_predict_probability=True, svm_class_weight="balanced"),
        TaskTrainSVM(max_vocab_size=100000, input_src_path=train_source, problem_type=ProblemType.RETURN_NONE, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.3, svm_kernel="rbf", svm_predict_probability=True, svm_class_weight="balanced"),

]

models_keras = [
        TaskTrainLstm(max_vocab_size=100000, input_src_path=train_source, problem_type=ProblemType.CONDITION_COMPARISON_SIMPLE, oversampling_enabled=False, undersampling_enabled=False, embedding_vecor_length=32, epochs=2, batch_size=256, encode_type=False, num_lstm_cells=10, dropout_emb_lstm=0.2, dropout_lstm_dense=0.2),
        TaskTrainLstm(max_vocab_size=100000, input_src_path=train_source, problem_type=ProblemType.CONDITION_COMPARISON_SIMPLE, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.5, embedding_vecor_length=32, epochs=3, batch_size=256, encode_type=False, dropout_emb_lstm=0.2, dropout_lstm_dense=0.2),
        TaskTrainLstm(max_vocab_size=100000, input_src_path=train_source, problem_type=ProblemType.CONDITION_COMPARISON_SIMPLE, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.5, embedding_vecor_length=32, epochs=3, batch_size=64, encode_type=False, num_lstm_cells=10, dropout_emb_lstm=0.2, dropout_lstm_dense=0.2),
        TaskTrainLstm(max_vocab_size=100000, input_src_path=train_source, problem_type=ProblemType.RETURN_NONE, oversampling_enabled=False, undersampling_enabled=False, embedding_vecor_length=32, epochs=2, batch_size=256, encode_type=False, num_lstm_cells=10, dropout_emb_lstm=0.2, dropout_lstm_dense=0.2),
        TaskTrainLstm(max_vocab_size=100000, input_src_path=train_source, problem_type=ProblemType.RETURN_NONE, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.5, embedding_vecor_length=32, epochs=3, batch_size=256, encode_type=False, dropout_emb_lstm=0.2, dropout_lstm_dense=0.2),
        TaskTrainLstm(max_vocab_size=100000, input_src_path=train_source, problem_type=ProblemType.RETURN_NONE, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.5, embedding_vecor_length=32, epochs=3, batch_size=64, encode_type=False, num_lstm_cells=10, dropout_emb_lstm=0.2, dropout_lstm_dense=0.2),
]

for model_task in models_ensemble:
    model = model_task.outputLoad()
    problem_type = model_task.problem_type
    if problem_type == ProblemType.CONDITION_COMPARISON_SIMPLE:
            problem_type = ProblemType.CONDITION_COMPARISON #using the trained model on simple to predict not simple stuff
    t = TaskEvalEnsemble(model=model, input_src_path=validation_source, problem_type=problem_type, vocab_input_directory=train_source, max_vocab_size=100000, encode_type=model_task.encode_type, undersampling_enabled=True, undersampling_ratio=0.1, training_parameter={**model_task.__dict__["param_kwargs"], "task_id": model_task.task_id})
    d6tflow.run(t, workers=2)

for model_task in models_keras:
    model = model_task.outputLoad()
    problem_type = model_task.problem_type
    if problem_type == ProblemType.CONDITION_COMPARISON_SIMPLE:
            problem_type = ProblemType.CONDITION_COMPARISON #using the trained model on simple to predict not simple stuff
    t = TaskEvalKeras(model=model, input_src_path=validation_source, problem_type=problem_type, vocab_input_directory=train_source, max_vocab_size=100000, encode_type=model_task.encode_type, undersampling_enabled=True, undersampling_ratio=0.1, training_parameter={**model_task.__dict__["param_kwargs"], "task_id": model_task.task_id})
    d6tflow.run(t, workers=1)




