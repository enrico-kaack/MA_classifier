import pandas as pd
import d6tflow

pd.set_option('display.max_colwidth', None)

from models.random_forst import decode_vector
from tasks.gradient_boosting_classifier import TaskEvaluateGradientBoostingClassifier, TaskTrainGradientBoostingClassifier
from tasks.random_forest import TaskEvaluateRandomForest, TaskTrainRandomForest
from tasks.lstm import TaskEvaluateLstm, TaskTrainLstm
from tasks.preprocessing import ProblemType, TaskVocabCreator
from tasks.holdout_test import TaskEvalHoldoutKeras, TaskEvalHoldoutEnsemble
from tasks.manipulate_code import TaskEvalEnsemble, TaskEvalKeras


#data = TaskEvaluateGradientBoostingClassifier(problem_type=ProblemType.CONDITION_COMPARISON_SIMPLE, oversampling_enabled=False, undersampling_enabled=False, learning_rate=0.2, n_estimators=300, subsample=1.0).outputLoad()
#data = TaskEvaluateRandomForest(problem_type=ProblemType.CONDITION_COMPARISON_SIMPLE, oversampling_enabled=True, ratio_after_oversampling=0.5, undersampling_enabled=False, encode_type=True, class_weight=None).outputLoad()
#data = TaskEvaluateRandomForest(problem_type=ProblemType.RETURN_NONE, oversampling_enabled=False, ratio_after_oversampling=0.5, undersampling_enabled=True, ratio_after_undersampling=0.5, encode_type=True, class_weight=None).outputLoad()
#encode_type = True
#data = TaskEvaluateLstm(problem_type=ProblemType.CONDITION_COMPARISON_SIMPLE, oversampling_enabled=False, undersampling_enabled=False, epochs=2, batch_size=256,num_lstm_cells=10, encode_type=False, dropout_emb_lstm=0.2, dropout_lstm_dense=0.2).outputLoad()
encode_type = False


model_task = TaskTrainLstm(problem_type=ProblemType.CONDITION_COMPARISON_SIMPLE, oversampling_enabled=False, undersampling_enabled=False, epochs=2, batch_size=256,num_lstm_cells=10, encode_type=False, dropout_emb_lstm=0.2, dropout_lstm_dense=0.2)
model = model_task.outputLoad()
problem_type = model_task.problem_type
t = TaskEvalHoldoutKeras(model=model, test_input_directory="final_validation", problem_type=problem_type, encode_type=model_task.encode_type, training_parameter={**model_task.__dict__["param_kwargs"], "task_id": model_task.task_id})
d6tflow.run(t)
data = t.outputLoad()


vocab = TaskVocabCreator().outputLoad()

#reverse vocab
reverse_vocab = {value:key for key, value in vocab.items()} 


def decode(row):
    unformated =  decode_vector(row["x"], reverse_vocab, encode_type)
    values = [v for i,v in enumerate(unformated) if i % 2] if encode_type else [v for i,v in enumerate(unformated)] 
    types = [v for i,v in enumerate(unformated) if not i % 2] if encode_type else []
    return [values, types]

def get_decoded_data(data):
    data["decoded"] =  data.apply(decode, axis=1)
    return data

def get_false_positive(data):
    data = data[(data["ground_truth"] == 0) & (data["predicted"] == 1)]
    return get_decoded_data(data)

def get_true_positive(data):
    data = data[(data["ground_truth"] == 1) & (data["predicted"] == 1)]
    return get_decoded_data(data)

def get_false_negative(data):
    data = data[(data["ground_truth"] == 1) & (data["predicted"] == 0)]
    return get_decoded_data(data)

def get_true_negative(data):
    data = data[(data["ground_truth"] == 0) & (data["predicted"] == 0)]
    return get_decoded_data(data)




with open("best_lstm_CCS_holdout.txt", "w") as output:

    values = {"tp": get_true_positive(data.copy()).sample(20)["decoded"].values, "fp": get_false_positive(data.copy()).sample(20)["decoded"].values, "tn": get_true_negative(data.copy()).sample(20)["decoded"].values, "fn": get_false_negative(data.copy()).sample(20)["decoded"].values}

    for key,val in values.items():
        output.write(key.upper() + "\n")
        for v in val:
            output.write(" ".join(v[0]))
            output.write("\n" +",".join(v[1]))
            output.write("\n########################\n")