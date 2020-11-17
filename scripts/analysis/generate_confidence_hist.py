import pandas as pd
import d6tflow
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

pd.set_option('display.max_colwidth', None)

from preprocessing.helper import decode_vector
from tasks.gradient_boosting_classifier import TaskEvaluateGradientBoostingClassifier
from tasks.random_forest import TaskEvaluateRandomForest
from tasks.lstm import TaskEvaluateLstm, TaskTrainLstm
from tasks.preprocessing import ProblemType, TaskVocabCreator
from tasks.holdout_test import TaskEvalHoldoutKeras

"""
model_task = TaskTrainLstm(problem_type=ProblemType.RETURN_NONE, oversampling_enabled=False, undersampling_enabled=True, ratio_after_undersampling=0.5, embedding_vecor_length=32, epochs=3, batch_size=256, encode_type=False, num_lstm_cells=10, dropout_emb_lstm=0.2, dropout_lstm_dense=0.2)

model = model_task.outputLoad()
problem_type = model_task.problem_type
eval_task = TaskEvalHoldoutKeras(model=model, problem_type=problem_type, encode_type=model_task.encode_type, training_parameter={**model_task.__dict__["param_kwargs"], "task_id": model_task.task_id})
d6tflow.run(eval_task)

model_task = TaskTrainLstm(problem_type=ProblemType.RETURN_NONE, oversampling_enabled=False, undersampling_enabled=False, ratio_after_undersampling=0.5, embedding_vecor_length=32, epochs=3, batch_size=256, encode_type=False, num_lstm_cells=10, dropout_emb_lstm=0.2, dropout_lstm_dense=0.2)

model = model_task.outputLoad()
problem_type = model_task.problem_type
eval_task = TaskEvalHoldoutKeras(model=model, problem_type=problem_type, encode_type=model_task.encode_type, training_parameter={**model_task.__dict__["param_kwargs"], "task_id": model_task.task_id})
d6tflow.run(eval_task)
print(eval_task.task_id)
"""
exit

def get_false_positive(data):
    data = data[(data["ground_truth"] == 0) & (data["predicted"] == 1)]
    return data

def get_true_positive(data):
    data = data[(data["ground_truth"] == 1) & (data["predicted"] == 1)]
    return data

def get_false_negative(data):
    data = data[(data["ground_truth"] == 1) & (data["predicted"] == 0)]
    return data

def get_true_negative(data):
    data = data[(data["ground_truth"] == 0) & (data["predicted"] == 0)]
    return data


with open("/home/enrico/UniProjects/Masterarbeit/own_classifier/data/TaskEvalHoldoutKeras/TaskEvalHoldoutKeras_False_100000__tensorflow_pyth_268a98039f-data.pkl", "rb") as input:
    results = pickle.load(input)
    tp = [x[0] for x in get_true_positive(results)["probability"].tolist()]
    fp = [x[0] for x in get_false_positive(results)["probability"].tolist()]

    tn = [x[0] for x in get_true_negative(results)["probability"].tolist()]
    fn = [x[0] for x in get_false_negative(results)["probability"].tolist()]

    fig, ax = plt.subplots()
    kwargs = dict(alpha=0.7, bins=50, density=True, stacked=False)
    ax.hist(tp, **kwargs, color='tab:blue', label='TP')
    ax.hist(fp, **kwargs, color='tab:red', label='FP')
    ax.hist(tn, **kwargs, color='tab:orange', label='TN')
    ax.hist(fn, **kwargs, color='tab:purple', label='FN')
    plt.legend()
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Number of samples")

    plt.savefig("rq2_lstm_undersampled_confidence_histogram.pgf")



with open("/home/enrico/UniProjects/Masterarbeit/own_classifier/data/TaskEvalHoldoutKeras/TaskEvalHoldoutKeras_False_100000__tensorflow_pyth_cb71c49df9-data.pkl", "rb") as input:
    results = pickle.load(input)
    tp = [x[0] for x in get_true_positive(results)["probability"].tolist()]
    fp = [x[0] for x in get_false_positive(results)["probability"].tolist()]

    tn = [x[0] for x in get_true_negative(results)["probability"].tolist()]
    fn = [x[0] for x in get_false_negative(results)["probability"].tolist()]

    fig, ax = plt.subplots()
    kwargs = dict(alpha=0.7, bins=50, density=True, stacked=False)
    ax.hist(tp, **kwargs, color='tab:blue', label='TP')
    ax.hist(fp, **kwargs, color='tab:red', label='FP')
    ax.hist(tn, **kwargs, color='tab:orange', label='TN')
    ax.hist(fn, **kwargs, color='tab:purple', label='FN')
    plt.legend()
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Number of samples")

    plt.savefig("rq2_lstm_not_resampled_confidence_histogram.pgf")
