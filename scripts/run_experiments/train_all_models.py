import argparse, sys
from tasks.preprocessing import ProblemType
from tasks.gradient_boosting_classifier import TaskEvaluateGradientBoostingClassifier
from tasks.svm import TaskEvaluateSVM
from tasks.lstm import TaskEvaluateLstm
from tasks.random_forest import TaskEvaluateRandomForest
import d6tflow

def run_all_tasks(source, workers):
    task_list= []

    for problem_type in ProblemType:
        t =[ 
            # #Random Forest
            # TaskEvaluateRandomForest(problem_type=problem_type, oversampling_enabled=False, undersampling_enabled=False, encode_type=True),
            # TaskEvaluateRandomForest(problem_type=problem_type, oversampling_enabled=False, undersampling_enabled=False, encode_type=False),
            # TaskEvaluateRandomForest(problem_type=problem_type, oversampling_enabled=False, undersampling_enabled=False, encode_type=True, class_weight="balanced"),
            # TaskEvaluateRandomForest(problem_type=problem_type, oversampling_enabled=False, undersampling_enabled=False, encode_type=False, class_weight="balanced"),

            # TaskEvaluateRandomForest(problem_type=problem_type, oversampling_enabled=True, ratio_after_oversampling=0.5, undersampling_enabled=False, encode_type=False),
            # TaskEvaluateRandomForest(problem_type=problem_type, oversampling_enabled=True, ratio_after_oversampling=0.5, undersampling_enabled=False, encode_type=True),
            # TaskEvaluateRandomForest(problem_type=problem_type, oversampling_enabled=True, ratio_after_oversampling=0.5, undersampling_enabled=False, encode_type=True, class_weight="balanced"),
            # TaskEvaluateRandomForest(problem_type=problem_type, oversampling_enabled=True, ratio_after_oversampling=1.0, undersampling_enabled=False, encode_type=False),
            # TaskEvaluateRandomForest(problem_type=problem_type, oversampling_enabled=True, ratio_after_oversampling=1.0, undersampling_enabled=False, encode_type=True),

            # TaskEvaluateRandomForest(problem_type=problem_type, oversampling_enabled=False, ratio_after_undersampling=0.5, undersampling_enabled=True, encode_type=False),
            # TaskEvaluateRandomForest(problem_type=problem_type, oversampling_enabled=False, ratio_after_undersampling=0.5, undersampling_enabled=True, encode_type=True),
            # TaskEvaluateRandomForest(problem_type=problem_type, oversampling_enabled=False, ratio_after_undersampling=0.5, undersampling_enabled=True, encode_type=True, class_weight="balanced"),
            # TaskEvaluateRandomForest(problem_type=problem_type, oversampling_enabled=False, ratio_after_undersampling=0.1, undersampling_enabled=True, encode_type=True),

            # # Gradient Boosting classifier
            # TaskEvaluateGradientBoostingClassifier(problem_type=problem_type, oversampling_enabled=False, undersampling_enabled=False, learning_rate=0.2, n_estimators=100, subsample=1.0),
            # TaskEvaluateGradientBoostingClassifier(problem_type=problem_type, oversampling_enabled=False, undersampling_enabled=False, learning_rate=0.2, n_estimators=300, subsample=1.0),
            # TaskEvaluateGradientBoostingClassifier(problem_type=problem_type, oversampling_enabled=False, undersampling_enabled=False, learning_rate=0.2, n_estimators=200, subsample=1.0),
            # TaskEvaluateGradientBoostingClassifier(problem_type=problem_type, oversampling_enabled=False, undersampling_enabled=False, learning_rate=0.2, n_estimators=200, subsample=0.4),
            # TaskEvaluateGradientBoostingClassifier(problem_type=problem_type, oversampling_enabled=False, undersampling_enabled=False, learning_rate=0.2, n_estimators=200, subsample=0.7),
            
            # TaskEvaluateGradientBoostingClassifier(problem_type=problem_type, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.5, learning_rate=0.2, n_estimators=200, subsample=1.0),
            # TaskEvaluateGradientBoostingClassifier(problem_type=problem_type, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.5, learning_rate=0.2, n_estimators=100, subsample=1.0),
            # TaskEvaluateGradientBoostingClassifier(problem_type=problem_type, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.5, learning_rate=0.1, n_estimators=100, subsample=1.0),

            # TaskEvaluateGradientBoostingClassifier(problem_type=problem_type, oversampling_enabled=True, undersampling_enabled=False, ratio_after_oversampling=0.5, learning_rate=0.2, n_estimators=200, subsample=1.0),
            # TaskEvaluateGradientBoostingClassifier(problem_type=problem_type, oversampling_enabled=True, undersampling_enabled=False, ratio_after_oversampling=0.5, learning_rate=0.2, n_estimators=100, subsample=1.0),
            # TaskEvaluateGradientBoostingClassifier(problem_type=problem_type, oversampling_enabled=True, undersampling_enabled=False, ratio_after_oversampling=0.5, learning_rate=0.1, n_estimators=100, subsample=1.0),

            # #LSTM
            # TaskEvaluateLstm(problem_type=problem_type, oversampling_enabled=False, undersampling_enabled=False, embedding_vecor_length=32, epochs=2, batch_size=256, encode_type=False, num_lstm_cells=10, dropout_emb_lstm=0.2, dropout_lstm_dense=0.2),
            # TaskEvaluateLstm(problem_type=problem_type, oversampling_enabled=False, undersampling_enabled=False, embedding_vecor_length=32, epochs=2, batch_size=512, encode_type=False, num_lstm_cells=10, dropout_emb_lstm=0.2, dropout_lstm_dense=0.2),
            # TaskEvaluateLstm(problem_type=problem_type, oversampling_enabled=False, undersampling_enabled=False,  embedding_vecor_length=32, epochs=3, batch_size=256, encode_type=False, num_lstm_cells=10, dropout_emb_lstm=0.2, dropout_lstm_dense=0.2),
            # TaskEvaluateLstm(problem_type=problem_type, oversampling_enabled=False, undersampling_enabled=False,  embedding_vecor_length=32, epochs=3, batch_size=512, encode_type=False, num_lstm_cells=10, dropout_emb_lstm=0.2, dropout_lstm_dense=0.2),
            # TaskEvaluateLstm(problem_type=problem_type, oversampling_enabled=False, undersampling_enabled=False, embedding_vecor_length=32, epochs=3, batch_size=64, encode_type=False, num_lstm_cells=10, dropout_emb_lstm=0.2, dropout_lstm_dense=0.2),

            # TaskEvaluateLstm(problem_type=problem_type, oversampling_enabled=False, undersampling_enabled=False, embedding_vecor_length=16, epochs=2, batch_size=256, encode_type=False, num_lstm_cells=10, dropout_emb_lstm=0.2, dropout_lstm_dense=0.2),
            # TaskEvaluateLstm(problem_type=problem_type, oversampling_enabled=False, undersampling_enabled=False, embedding_vecor_length=16, epochs=3, batch_size=256, encode_type=False, num_lstm_cells=10, dropout_emb_lstm=0.2, dropout_lstm_dense=0.2),

            TaskEvaluateLstm(problem_type=problem_type, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.5, embedding_vecor_length=32, epochs=3, batch_size=256, encode_type=False, num_lstm_cells=10, dropout_emb_lstm=0.2, dropout_lstm_dense=0.2),
            TaskEvaluateLstm(problem_type=problem_type, oversampling_enabled=True, undersampling_enabled=False, ratio_after_oversampling=0.5, embedding_vecor_length=32, epochs=3, batch_size=256, encode_type=False, num_lstm_cells=10, dropout_emb_lstm=0.2, dropout_lstm_dense=0.2)
        ]

        task_list.extend(t)



    d6tflow.preview(task_list)
    d6tflow.run(task_list, workers=workers)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='train_all_models.py', description="Run all tasks for given problem type")
    parser.add_argument("source", help="src folder name", metavar="SOURCE")
    parser.add_argument("-n", help="number of workers to use", metavar="WORKERS")

    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    source = args.source
    workers = args.n
    run_all_tasks(source, workers)

#python -m scripts.run_experiments.train_all_models final_dataset -n 4