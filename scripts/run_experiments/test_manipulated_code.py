import argparse, sys
from tasks.preprocessing import ProblemType
from tasks.gradient_boosting_classifier import TaskTrainGradientBoostingClassifier
from tasks.svm import TaskTrainSVM
from tasks.lstm import TaskTrainLstm
from tasks.random_forest import TaskTrainRandomForest
from tasks.manipulate_code import TaskEvalEnsemble, TaskEvalKeras
import d6tflow

def run_all_tasks(validation_source, workers):
    task_list_ensemble = []
    task_list_keras = []

    for problem_type in [ProblemType.RETURN_NONE, ProblemType.CONDITION_COMPARISON_SIMPLE, ProblemType.CONDITION_COMPARISON]:
        t =[ 
            #Random Forest
            TaskTrainRandomForest(problem_type=problem_type, oversampling_enabled=False, undersampling_enabled=False, encode_type=True),
            TaskTrainRandomForest(problem_type=problem_type, oversampling_enabled=False, undersampling_enabled=False, encode_type=False),
            TaskTrainRandomForest(problem_type=problem_type, oversampling_enabled=False, undersampling_enabled=False, encode_type=True, class_weight="balanced"),
            TaskTrainRandomForest(problem_type=problem_type, oversampling_enabled=False, undersampling_enabled=False, encode_type=False, class_weight="balanced"),

            TaskTrainRandomForest(problem_type=problem_type, oversampling_enabled=True, ratio_after_oversampling=0.5, undersampling_enabled=False, encode_type=False),
            TaskTrainRandomForest(problem_type=problem_type, oversampling_enabled=True, ratio_after_oversampling=0.5, undersampling_enabled=False, encode_type=True),
            TaskTrainRandomForest(problem_type=problem_type, oversampling_enabled=True, ratio_after_oversampling=0.5, undersampling_enabled=False, encode_type=True, class_weight="balanced"),
            TaskTrainRandomForest(problem_type=problem_type, oversampling_enabled=True, ratio_after_oversampling=1.0, undersampling_enabled=False, encode_type=False),
            TaskTrainRandomForest(problem_type=problem_type, oversampling_enabled=True, ratio_after_oversampling=1.0, undersampling_enabled=False, encode_type=True),

            TaskTrainRandomForest(problem_type=problem_type, oversampling_enabled=False, ratio_after_undersampling=0.5, undersampling_enabled=True, encode_type=False),
            TaskTrainRandomForest(problem_type=problem_type, oversampling_enabled=False, ratio_after_undersampling=0.5, undersampling_enabled=True, encode_type=True),
            TaskTrainRandomForest(problem_type=problem_type, oversampling_enabled=False, ratio_after_undersampling=0.5, undersampling_enabled=True, encode_type=True, class_weight="balanced"),
            TaskTrainRandomForest(problem_type=problem_type, oversampling_enabled=False, ratio_after_undersampling=0.1, undersampling_enabled=True, encode_type=True),

            # Gradient Boosting classifier
            TaskTrainGradientBoostingClassifier(problem_type=problem_type, oversampling_enabled=False, undersampling_enabled=False, learning_rate=0.2, n_estimators=100, subsample=1.0),
            TaskTrainGradientBoostingClassifier(problem_type=problem_type, oversampling_enabled=False, undersampling_enabled=False, learning_rate=0.2, n_estimators=300, subsample=1.0),
            TaskTrainGradientBoostingClassifier(problem_type=problem_type, oversampling_enabled=False, undersampling_enabled=False, learning_rate=0.2, n_estimators=200, subsample=1.0),
            TaskTrainGradientBoostingClassifier(problem_type=problem_type, oversampling_enabled=False, undersampling_enabled=False, learning_rate=0.2, n_estimators=200, subsample=0.4),
            TaskTrainGradientBoostingClassifier(problem_type=problem_type, oversampling_enabled=False, undersampling_enabled=False, learning_rate=0.2, n_estimators=200, subsample=0.7),
            
            TaskTrainGradientBoostingClassifier(problem_type=problem_type, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.5, learning_rate=0.2, n_estimators=200, subsample=1.0),
            TaskTrainGradientBoostingClassifier(problem_type=problem_type, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.5, learning_rate=0.2, n_estimators=100, subsample=1.0),
            TaskTrainGradientBoostingClassifier(problem_type=problem_type, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.5, learning_rate=0.1, n_estimators=100, subsample=1.0),

            TaskTrainGradientBoostingClassifier(problem_type=problem_type, oversampling_enabled=True, undersampling_enabled=False, ratio_after_oversampling=0.5, learning_rate=0.2, n_estimators=200, subsample=1.0),
            TaskTrainGradientBoostingClassifier(problem_type=problem_type, oversampling_enabled=True, undersampling_enabled=False, ratio_after_oversampling=0.5, learning_rate=0.2, n_estimators=100, subsample=1.0),
            TaskTrainGradientBoostingClassifier(problem_type=problem_type, oversampling_enabled=True, undersampling_enabled=False, ratio_after_oversampling=0.5, learning_rate=0.1, n_estimators=100, subsample=1.0),

        ]
        task_list_ensemble.extend(t)
        t= [            
            #LSTM
            TaskTrainLstm(problem_type=problem_type, oversampling_enabled=False, undersampling_enabled=False, embedding_vecor_length=32, epochs=2, batch_size=256, encode_type=False, num_lstm_cells=10, dropout_emb_lstm=0.2, dropout_lstm_dense=0.2),
            TaskTrainLstm(problem_type=problem_type, oversampling_enabled=False, undersampling_enabled=False, embedding_vecor_length=32, epochs=2, batch_size=512, encode_type=False, num_lstm_cells=10, dropout_emb_lstm=0.2, dropout_lstm_dense=0.2),
            TaskTrainLstm(problem_type=problem_type, oversampling_enabled=False, undersampling_enabled=False,  embedding_vecor_length=32, epochs=3, batch_size=256, encode_type=False, num_lstm_cells=10, dropout_emb_lstm=0.2, dropout_lstm_dense=0.2),
            TaskTrainLstm(problem_type=problem_type, oversampling_enabled=False, undersampling_enabled=False,  embedding_vecor_length=32, epochs=3, batch_size=512, encode_type=False, num_lstm_cells=10, dropout_emb_lstm=0.2, dropout_lstm_dense=0.2),
            TaskTrainLstm(problem_type=problem_type, oversampling_enabled=False, undersampling_enabled=False, embedding_vecor_length=32, epochs=3, batch_size=64, encode_type=False, num_lstm_cells=10, dropout_emb_lstm=0.2, dropout_lstm_dense=0.2),

            TaskTrainLstm(problem_type=problem_type, oversampling_enabled=False, undersampling_enabled=False, embedding_vecor_length=16, epochs=2, batch_size=256, encode_type=False, num_lstm_cells=10, dropout_emb_lstm=0.2, dropout_lstm_dense=0.2),
            TaskTrainLstm(problem_type=problem_type, oversampling_enabled=False, undersampling_enabled=False, embedding_vecor_length=16, epochs=3, batch_size=256, encode_type=False, num_lstm_cells=10, dropout_emb_lstm=0.2, dropout_lstm_dense=0.2),

            TaskTrainLstm(problem_type=problem_type, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.5, embedding_vecor_length=32, epochs=3, batch_size=256, encode_type=False,num_lstm_cells=10, dropout_emb_lstm=0.2, dropout_lstm_dense=0.2),
            TaskTrainLstm(problem_type=problem_type, oversampling_enabled=True, undersampling_enabled=False, ratio_after_oversampling=0.5, embedding_vecor_length=32, epochs=3, batch_size=256, encode_type=False,num_lstm_cells=10, dropout_emb_lstm=0.2, dropout_lstm_dense=0.2)
        ]
        task_list_keras.extend(t)


    final_tasks = []
    for model_task in task_list_ensemble:
        model = model_task.outputLoad()
        problem_type = model_task.problem_type
        if problem_type == ProblemType.CONDITION_COMPARISON_SIMPLE:
                problem_type = ProblemType.CONDITION_COMPARISON #using the trained model on simple to predict not simple stuff
        t = TaskEvalEnsemble(model=model, test_input_directory=validation_source, problem_type=problem_type, encode_type=model_task.encode_type, training_parameter={**model_task.__dict__["param_kwargs"], "task_id": model_task.task_id})
        final_tasks.append(t)

    for model_task in task_list_keras:
        model = model_task.outputLoad()
        problem_type = model_task.problem_type
        if problem_type == ProblemType.CONDITION_COMPARISON_SIMPLE:
                problem_type = ProblemType.CONDITION_COMPARISON #using the trained model on simple to predict not simple stuff
        t = TaskEvalKeras(model=model, test_input_directory=validation_source, problem_type=problem_type, encode_type=model_task.encode_type, training_parameter={**model_task.__dict__["param_kwargs"], "task_id": model_task.task_id})
        final_tasks.append(t)
    d6tflow.preview(final_tasks)
    d6tflow.run(final_tasks, workers=workers)






if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='test_manipulated_code.py', description="Evaluate the models on manipulated code")
    parser.add_argument("source", help="src folder name of validation set", metavar="SOURCE")
    parser.add_argument("-n", help="number of workers to use", metavar="WORKERS")

    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    source = args.source
    workers = args.n
    run_all_tasks(source, workers)

#python -m scripts.run_experiments.test_manipulated_code final_validation -n 4