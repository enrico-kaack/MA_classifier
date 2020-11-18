import os
import jinja2
import pandas as pd
import json
import argparse
import re
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

#generate plot for figure 4.2


def process_data(results, model):
    results["ratio_after_oversampling"][results.oversampling_enabled == False] = "-"
    results["ratio_after_undersampling"][results.undersampling_enabled == False] = "-"


    #check for right dataset run
    results = results[results["max_vocab_size"] == 100000]

    all_merge_columns = {"Random Forest": ['window_size', 'step_size', 'encode_type','oversampling_enabled', 'ratio_after_oversampling', 'undersampling_enabled', 'ratio_after_undersampling', 'n_trees_in_forest', 'max_features', 'class_weight', 'encode_type'],

    "Gradient Boosting Classifier": ['oversampling_enabled', 'ratio_after_oversampling', 'undersampling_enabled', 'ratio_after_undersampling', 'n_estimators', 'learning_rate', 'subsample', 'encode_type'],

    "LSTM": ['oversampling_enabled', 'ratio_after_oversampling', 'undersampling_enabled', 'ratio_after_undersampling','embedding_vecor_length', 'epochs', 'batch_size', 'num_lstm_cells', 'dropout_emb_lstm', 'dropout_lstm_dense', 'encode_type'],

    "svm": ['oversampling_enabled', 'ratio_after_oversampling', 'undersampling_enabled', 'ratio_after_undersampling','svm_kernel', 'svm_subsample', 'svm_class_weight', 'encode_type']
    }
    merge_columns = all_merge_columns[model]

    all_filter = {"Random Forest": "n_trees_in_forest", "Gradient Boosting Classifier": "n_estimators", "LSTM": "embedding_vecor_length", "svm": "svm_kernel"}
    model_filter = all_filter[model]

    random_forest_RN = results[(results["problem_type"] == "RETURN_NULL")  & (results[model_filter].notnull())]
    random_forest_RN.columns = random_forest_RN.columns.map(lambda a: a + "_RN" if a not in merge_columns else a)

    random_forest_CC = results[(results["problem_type"] == "CONDITION_COMPARISON")  & (results[model_filter].notnull())]
    random_forest_CC.columns = random_forest_CC.columns.map(lambda a: a + "_CC" if a not in merge_columns else a)

    random_forest_CCS = results[(results["problem_type"] == "CONDITION_COMPARISON_SIMPLE")  & (results[model_filter].notnull())]
    random_forest_CCS.columns = random_forest_CCS.columns.map(lambda a: a + "_CCS" if a not in merge_columns else a)


    merged = random_forest_RN.merge(random_forest_CC, on=merge_columns, suffixes=("_LEFT1", "_RIGHT1"), how="outer").merge(random_forest_CCS, on=merge_columns, suffixes=("_LEFT2", "_RIGHT2"), how="outer")

    merged = merged.sort_values(by="manipulated_f1_CC", ascending=False)
    #print(list(merged.columns))
    return (model, merged)

def read_results(input_dir):
    frames = []
    for root, dirs, files in os.walk(input_dir):
        for name in files:
            if name.endswith(".json"):
                with open(os.path.join(root, name), "r") as data_file:
                    data = json.load(data_file)
                    frame = pd.json_normalize(data, sep=".")
                    frames.append(frame)
    results = pd.concat(frames)
    results = results.drop(columns=["task_id", "window_size", "step_size", "encode_type", "vocab_input_directory", "test_input_directory", "max_vocab_size", "problem_type"])
    results = results.rename(columns=lambda x: re.sub('training_parameter.','',x))
    return results

def print_data(data):
    rendered = []
    fig, ax = plt.subplots(3)
    fig.suptitle("F1 Performance Comparison")
    index = 0
    p = []
    for model, d in data:
        values = d.T.to_dict().values()

        baseline = [x["manipulated_f1_CC"] for x in values]
        ccs_values = [x["manipulated_f1_CCS"] for x in values]
        rn_values = [x["manipulated_f1_RN"] for x in values]

        N = len(baseline)


        ind = np.arange(N)    # the x locations for the groups
        width = 0.2         # the width of the bars
        p1 = ax[index].bar(ind, baseline, width, bottom=0)
        p2 = ax[index].bar(ind + width, ccs_values, width, bottom=0)
        p3 = ax[index].bar(ind + 2*width, rn_values, width, bottom=0)
        #p1 = ax[index].plot(baseline)
        #p2 = ax[index].plot(ccs_values)
        #p3 = ax[index].plot(rn_values)

        p.extend([p1, p2, p3])
        #if index == 0:
        #    ax[index].legend((p1[0], p2[0], p3[0]), ('Baseline', 'manipulated CCS', 'manipulated RN'), loc=1)

        ax[index].set_title(f"{model}")

        ax[index].set_xticks(ind + width + width / 2)
        ax[index].set_xticklabels(ind)
        ax[index].set_ylabel("F1")
        ax[index].set_yticks([0,0.5,1])


        index += 1
    
    fig.legend(p, ['Baseline for CCS', 'Manipulated CCS', 'Manipulated RN'],  bbox_to_anchor=(0.5, -0.01), loc='lower center')

    plt.subplots_adjust(hspace=0.7, bottom=0.2)
    #plt.show()

    plt.savefig(f"rq3_performance_comparison.pgf")





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='genereate bar chart with sorted models and performance (see figure 4.2)')
    parser.add_argument('dir', type=str, help='Input dir results for manipulated code experiment')
    args = parser.parse_args()

    results = read_results(args.dir)
    data = []
    for model in ["Random Forest", "Gradient Boosting Classifier", "LSTM"]:
        data.append(process_data(results, model))
    print_data(data)
