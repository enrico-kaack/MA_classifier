import pandas as pd
import d6tflow

pd.set_option('display.max_colwidth', None)

from models.random_forst import decode_vector
from tasks.gradient_boosting_classifier import TaskEvaluateGradientBoostingClassifier
from tasks.preprocessing import ProblemType, TaskVocabCreator

data = TaskEvaluateGradientBoostingClassifier(problem_type=ProblemType.CONDITION_COMPARISON_SIMPLE, oversampling_enabled=False, undersampling_enabled=False, learning_rate=0.2, n_estimators=300, subsample=1.0).outputLoad()
vocab = TaskVocabCreator().outputLoad()

#reverse vocab
reverse_vocab = {value:key for key, value in vocab.items()} 
encode_type = True


def decode(row):
    unformated =  decode_vector(row["x"], reverse_vocab)
    values = [v for i,v in enumerate(unformated) if not i % 2]
    types = [v for i,v in enumerate(unformated) if i % 2] if encode_type else None
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




with open("output.txt", "w") as output:

    values = {"tp": get_true_positive(data.copy()).sample(20)["decoded"].values, "fp": get_false_positive(data.copy()).sample(20)["decoded"].values, "tn": get_true_negative(data.copy()).sample(20)["decoded"].values, "fn": get_false_negative(data.copy()).sample(20)["decoded"].values}

    for key,val in values.items():
        output.write(key.upper() + "\n")
        for v in val:
            output.write(" ".join(v[1]))
            output.write("\n" +",".join(v[0]))
            output.write("\n########################\n")