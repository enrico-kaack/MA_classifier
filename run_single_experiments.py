from tasks.preprocessing import ProblemType, TaskVocabCreator
from tasks.gradient_boosting_classifier import TaskTrainGradientBoostingClassifier, TaskEvaluateGradientBoostingClassifier
from tasks.svm import TaskTrainSVM
from tasks.lstm import TaskTrainLstm
from tasks.random_forest import TaskTrainRandomForest
from tasks.manipulate_code import TaskEvalEnsemble, TaskEvalKeras, TaskPrepareXYValidation
import d6tflow

d6tflow.settings.log_level = 'INFO' # 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'

train_source = "second_large_dataset"
validation_source = "validation"

tasks_eval = []
models_ensemble = []
models_keras = []

for problem_type in ProblemType:
    t =[ 
        TaskEvaluateGradientBoostingClassifier(max_vocab_size=100000, input_src_path=train_source, problem_type=problem_type, undersampling_enabled=True, ratio_after_undersampling=0.5, n_estimators = 200, learning_rate=0.2),
        TaskEvaluateGradientBoostingClassifier(max_vocab_size=100000, input_src_path=train_source, problem_type=problem_type, undersampling_enabled=True, ratio_after_undersampling=0.5, n_estimators = 300, learning_rate=0.2),
        TaskEvaluateGradientBoostingClassifier(max_vocab_size=100000, input_src_path=train_source, problem_type=problem_type, undersampling_enabled=True, ratio_after_undersampling=0.5, n_estimators = 400, learning_rate=0.2),
        TaskEvaluateGradientBoostingClassifier(max_vocab_size=100000, input_src_path=train_source, problem_type=problem_type, undersampling_enabled=True, ratio_after_undersampling=0.5, n_estimators = 200, learning_rate=0.1),
        TaskEvaluateGradientBoostingClassifier(max_vocab_size=100000, input_src_path=train_source, problem_type=problem_type, undersampling_enabled=True, ratio_after_undersampling=0.5, n_estimators = 300, learning_rate=0.1),
        TaskEvaluateGradientBoostingClassifier(max_vocab_size=100000, input_src_path=train_source, problem_type=problem_type, undersampling_enabled=True, ratio_after_undersampling=0.5, n_estimators = 400, learning_rate=0.1)
    ]

    models_ensemble.extend([
        TaskTrainGradientBoostingClassifier(max_vocab_size=100000, input_src_path=train_source, problem_type=problem_type, undersampling_enabled=True, ratio_after_undersampling=0.5, n_estimators = 200, learning_rate=0.2),
        TaskTrainGradientBoostingClassifier(max_vocab_size=100000, input_src_path=train_source, problem_type=problem_type, undersampling_enabled=True, ratio_after_undersampling=0.5, n_estimators = 300, learning_rate=0.2),
        TaskTrainGradientBoostingClassifier(max_vocab_size=100000, input_src_path=train_source, problem_type=problem_type, undersampling_enabled=True, ratio_after_undersampling=0.5, n_estimators = 400, learning_rate=0.2),
        TaskTrainGradientBoostingClassifier(max_vocab_size=100000, input_src_path=train_source, problem_type=problem_type, undersampling_enabled=True, ratio_after_undersampling=0.5, n_estimators = 200, learning_rate=0.1),
        TaskTrainGradientBoostingClassifier(max_vocab_size=100000, input_src_path=train_source, problem_type=problem_type, undersampling_enabled=True, ratio_after_undersampling=0.5, n_estimators = 300, learning_rate=0.1),
        TaskTrainGradientBoostingClassifier(max_vocab_size=100000, input_src_path=train_source, problem_type=problem_type, undersampling_enabled=True, ratio_after_undersampling=0.5, n_estimators = 400, learning_rate=0.1)

    ])
    tasks_eval.extend(t)
d6tflow.run(tasks_eval, workers=2)



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
