import os
import jinja2
import pandas as pd
import json
import argparse
import re

latex_jinja_env = jinja2.Environment(
    block_start_string='\BLOCK{',
    block_end_string='}',
    variable_start_string='\VAR{',
    variable_end_string='}',
    comment_start_string='\#{',
    comment_end_string='}',
    line_statement_prefix='%%',
    line_comment_prefix='%#',
    trim_blocks=True,
    autoescape=False)

def process_data(results, model):
    results["ratio_after_oversampling"][results.oversampling_enabled == False] = "-"
    results["ratio_after_undersampling"][results.undersampling_enabled == False] = "-"


    #check for right dataset run
    results = results[results["max_vocab_size"] == 100000]

    all_merge_columns = {"random forest": ['window_size', 'step_size', 'encode_type','oversampling_enabled', 'ratio_after_oversampling', 'undersampling_enabled', 'ratio_after_undersampling', 'n_trees_in_forest', 'max_features', 'class_weight', 'encode_type'],

    "gradient boosting classifier": ['oversampling_enabled', 'ratio_after_oversampling', 'undersampling_enabled', 'ratio_after_undersampling', 'n_estimators', 'learning_rate', 'subsample', 'encode_type'],

    "lstm": ['oversampling_enabled', 'ratio_after_oversampling', 'undersampling_enabled', 'ratio_after_undersampling','embedding_vecor_length', 'epochs', 'batch_size', 'num_lstm_cells', 'dropout_emb_lstm', 'dropout_lstm_dense', 'encode_type'],

    "svm": ['oversampling_enabled', 'ratio_after_oversampling', 'undersampling_enabled', 'ratio_after_undersampling','svm_kernel', 'svm_subsample', 'svm_class_weight', 'encode_type']
    }
    merge_columns = all_merge_columns[model]

    all_filter = {"random forest": "n_trees_in_forest", "gradient boosting classifier": "n_estimators", "lstm": "embedding_vecor_length", "svm": "svm_kernel"}
    model_filter = all_filter[model]

    random_forest_RN = results[(results["problem_type"] == "RETURN_NULL")  & (results[model_filter].notnull())]
    random_forest_RN.columns = random_forest_RN.columns.map(lambda a: a + "_RN" if a not in merge_columns else a)

    random_forest_CC = results[(results["problem_type"] == "CONDITION_COMPARISON")  & (results[model_filter].notnull())]
    random_forest_CC.columns = random_forest_CC.columns.map(lambda a: a + "_CC" if a not in merge_columns else a)



    merged = random_forest_RN.merge(random_forest_CC, on=merge_columns, suffixes=("_LEFT1", "_RIGHT1"), how="outer")

    merged = merged.sort_values(by="manipulated_f1_RN", ascending=False)
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
    results = results.drop(columns=["task_id", "window_size", "step_size", "encode_type", "vocab_input_directory", "test_input_directory", "max_vocab_size", "training_parameter.problem_type"])
    results = results.rename(columns=lambda x: re.sub('training_parameter.','',x))
    return results

def print_data(data, output_file, template_file):
        rendered = []
        for model, d in data:
                with open(f"{template_file}.{model.replace(' ', '_')}", "r") as f:
                    t = latex_jinja_env.from_string(f.read())
                    rendered.append(t.render(data=d.T.to_dict().values(), model=model))
        with open(output_file, "w") as o:
            o.write("\n\n".join(rendered))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Write tex table for manipulate code results')
    parser.add_argument('dir', type=str, help='Input dir results')
    parser.add_argument('template', type=str, help='File for tempalte')
    parser.add_argument('output', type=str, help='Output File path')
    args = parser.parse_args()

    results = read_results(args.dir)
    data = []
    for model in ["random forest", "gradient boosting classifier", "lstm"]:
        data.append(process_data(results, model))
    print_data(data, args.output, args.template)
