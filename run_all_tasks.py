import argparse, sys
from tasks.preprocessing import ProblemType
from tasks.gradient_boosting_classifier import TaskEvaluateGradientBoostingClassifier
from tasks.svm import TaskEvaluateSVM
from tasks.lstm import TaskEvaluateLstm
from tasks.random_forest import TaskEvaluateRandomForest
import d6tflow

def run_all_tasks(source):
    task_list= [
        TaskEvaluateRandomForest(max_vocab_size=1000, input_src_path=source, problem_type=ProblemType.RETURN_NONE, oversampling_enabled=True, ratio_after_oversampling=0.5),
        TaskEvaluateRandomForest(max_vocab_size=1000, input_src_path=source, problem_type=ProblemType.RETURN_NONE, oversampling_enabled=True, ratio_after_oversampling=1.0),
        TaskEvaluateRandomForest(max_vocab_size=1000, input_src_path=source, problem_type=ProblemType.RETURN_NONE, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.5),
        TaskEvaluateRandomForest(max_vocab_size=1000, input_src_path=source, problem_type=ProblemType.RETURN_NONE, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.1),
        TaskEvaluateSVM(max_vocab_size=1000, input_src_path=source, problem_type=ProblemType.RETURN_NONE, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.5, svm_kernel="rbf", svm_predict_probability=True),
        TaskEvaluateSVM(max_vocab_size=1000, input_src_path=source, problem_type=ProblemType.RETURN_NONE, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.1, svm_kernel="rbf", svm_predict_probability=True),
        TaskEvaluateSVM(max_vocab_size=1000, input_src_path=source, problem_type=ProblemType.RETURN_NONE, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.5, svm_kernel="rbf", svm_predict_probability=True, svm_class_weight="balanced"),
        TaskEvaluateSVM(max_vocab_size=1000, input_src_path=source, problem_type=ProblemType.RETURN_NONE, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.3, svm_kernel="rbf", svm_predict_probability=True, svm_class_weight="balanced"),
        TaskEvaluateGradientBoostingClassifier(max_vocab_size=1000, input_src_path=source, problem_type=ProblemType.RETURN_NONE, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.5, learning_rate=0.1, n_estimators=100),
        TaskEvaluateGradientBoostingClassifier(max_vocab_size=1000, input_src_path=source, problem_type=ProblemType.RETURN_NONE, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.5, learning_rate=0.2, n_estimators=100),
        TaskEvaluateGradientBoostingClassifier(max_vocab_size=1000, input_src_path=source, problem_type=ProblemType.RETURN_NONE, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.5, learning_rate=0.01, n_estimators=1000),
        TaskEvaluateLstm(max_vocab_size=1000, input_src_path=source, problem_type=ProblemType.RETURN_NONE, oversampling_enabled=False, undersampling_enabled=False, embedding_vecor_length=32, epochs=2, batch_size=256, encode_type=False, num_lstm_cells=10, dropout_emb_lstm=0.2, dropout_lstm_dense=0.2),
        TaskEvaluateLstm(max_vocab_size=1000, input_src_path=source, problem_type=ProblemType.RETURN_NONE, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.5, embedding_vecor_length=32, epochs=3, batch_size=256, encode_type=False, dropout_emb_lstm=0.2, dropout_lstm_dense=0.2),
        TaskEvaluateLstm(max_vocab_size=1000, input_src_path=source, problem_type=ProblemType.RETURN_NONE, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.5, embedding_vecor_length=32, epochs=3, batch_size=64, encode_type=False, num_lstm_cells=10, dropout_emb_lstm=0.2, dropout_lstm_dense=0.2),

        TaskEvaluateRandomForest(max_vocab_size=100000, input_src_path=source, problem_type=ProblemType.RETURN_NONE, oversampling_enabled=True, ratio_after_oversampling=0.5),
        TaskEvaluateRandomForest(max_vocab_size=100000, input_src_path=source, problem_type=ProblemType.RETURN_NONE, oversampling_enabled=True, ratio_after_oversampling=1.0),
        TaskEvaluateRandomForest(max_vocab_size=100000, input_src_path=source, problem_type=ProblemType.RETURN_NONE, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.5),
        TaskEvaluateRandomForest(max_vocab_size=100000, input_src_path=source, problem_type=ProblemType.RETURN_NONE, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.1),
        TaskEvaluateSVM(max_vocab_size=100000, input_src_path=source, problem_type=ProblemType.RETURN_NONE, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.5, svm_kernel="rbf", svm_predict_probability=True),
        TaskEvaluateSVM(max_vocab_size=100000, input_src_path=source, problem_type=ProblemType.RETURN_NONE, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.1, svm_kernel="rbf", svm_predict_probability=True),
        TaskEvaluateSVM(max_vocab_size=100000, input_src_path=source, problem_type=ProblemType.RETURN_NONE, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.5, svm_kernel="rbf", svm_predict_probability=True, svm_class_weight="balanced"),
        TaskEvaluateSVM(max_vocab_size=100000, input_src_path=source, problem_type=ProblemType.RETURN_NONE, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.3, svm_kernel="rbf", svm_predict_probability=True, svm_class_weight="balanced"),
        TaskEvaluateGradientBoostingClassifier(max_vocab_size=100000, input_src_path=source, problem_type=ProblemType.RETURN_NONE, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.5, learning_rate=0.1, n_estimators=100),
        TaskEvaluateGradientBoostingClassifier(max_vocab_size=100000, input_src_path=source, problem_type=ProblemType.RETURN_NONE, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.5, learning_rate=0.2, n_estimators=100),
        TaskEvaluateGradientBoostingClassifier(max_vocab_size=100000, input_src_path=source, problem_type=ProblemType.RETURN_NONE, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.5, learning_rate=0.01, n_estimators=1000),
        TaskEvaluateLstm(max_vocab_size=100000, input_src_path=source, problem_type=ProblemType.RETURN_NONE, oversampling_enabled=False, undersampling_enabled=False, embedding_vecor_length=32, epochs=2, batch_size=256, encode_type=False, num_lstm_cells=10, dropout_emb_lstm=0.2, dropout_lstm_dense=0.2),
        TaskEvaluateLstm(max_vocab_size=100000, input_src_path=source, problem_type=ProblemType.RETURN_NONE, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.5, embedding_vecor_length=32, epochs=3, batch_size=256, encode_type=False, dropout_emb_lstm=0.2, dropout_lstm_dense=0.2),
        TaskEvaluateLstm(max_vocab_size=100000, input_src_path=source, problem_type=ProblemType.RETURN_NONE, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.5, embedding_vecor_length=32, epochs=3, batch_size=64, encode_type=False, num_lstm_cells=10, dropout_emb_lstm=0.2, dropout_lstm_dense=0.2),

        TaskEvaluateRandomForest(max_vocab_size=100000, input_src_path=source, problem_type=ProblemType.CONDITION_COMPARISON, oversampling_enabled=True, ratio_after_oversampling=0.5),
        TaskEvaluateRandomForest(max_vocab_size=100000, input_src_path=source, problem_type=ProblemType.CONDITION_COMPARISON, oversampling_enabled=True, ratio_after_oversampling=1.0),
        TaskEvaluateRandomForest(max_vocab_size=100000, input_src_path=source, problem_type=ProblemType.CONDITION_COMPARISON, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.5),
        TaskEvaluateRandomForest(max_vocab_size=100000, input_src_path=source, problem_type=ProblemType.CONDITION_COMPARISON, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.1),
        TaskEvaluateSVM(max_vocab_size=100000, input_src_path=source, problem_type=ProblemType.CONDITION_COMPARISON, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.5, svm_kernel="rbf", svm_predict_probability=True),
        TaskEvaluateSVM(max_vocab_size=100000, input_src_path=source, problem_type=ProblemType.CONDITION_COMPARISON, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.1, svm_kernel="rbf", svm_predict_probability=True),
        TaskEvaluateSVM(max_vocab_size=100000, input_src_path=source, problem_type=ProblemType.CONDITION_COMPARISON, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.5, svm_kernel="rbf", svm_predict_probability=True, svm_class_weight="balanced"),
        TaskEvaluateSVM(max_vocab_size=100000, input_src_path=source, problem_type=ProblemType.CONDITION_COMPARISON, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.3, svm_kernel="rbf", svm_predict_probability=True, svm_class_weight="balanced"),
        TaskEvaluateGradientBoostingClassifier(max_vocab_size=100000, input_src_path=source, problem_type=ProblemType.CONDITION_COMPARISON, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.5, learning_rate=0.1, n_estimators=100),
        TaskEvaluateGradientBoostingClassifier(max_vocab_size=100000, input_src_path=source, problem_type=ProblemType.CONDITION_COMPARISON, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.5, learning_rate=0.2, n_estimators=100),
        TaskEvaluateGradientBoostingClassifier(max_vocab_size=100000, input_src_path=source, problem_type=ProblemType.CONDITION_COMPARISON, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.5, learning_rate=0.01, n_estimators=1000),
        TaskEvaluateLstm(max_vocab_size=100000, input_src_path=source, problem_type=ProblemType.CONDITION_COMPARISON, oversampling_enabled=False, undersampling_enabled=False, embedding_vecor_length=32, epochs=2, batch_size=256, encode_type=False, num_lstm_cells=10, dropout_emb_lstm=0.2, dropout_lstm_dense=0.2),
        TaskEvaluateLstm(max_vocab_size=100000, input_src_path=source, problem_type=ProblemType.CONDITION_COMPARISON, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.5, embedding_vecor_length=32, epochs=3, batch_size=256, encode_type=False, dropout_emb_lstm=0.2, dropout_lstm_dense=0.2),
        TaskEvaluateLstm(max_vocab_size=100000, input_src_path=source, problem_type=ProblemType.CONDITION_COMPARISON, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.5, embedding_vecor_length=32, epochs=3, batch_size=64, encode_type=False, num_lstm_cells=10, dropout_emb_lstm=0.2, dropout_lstm_dense=0.2)
    ]
    d6tflow.run(task_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='run_all_tasks.py', description="Run all tasks for given problem type")
    parser.add_argument("source", help="src folder name", metavar="SOURCE")
    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    source = args.source
    run_all_tasks(source)