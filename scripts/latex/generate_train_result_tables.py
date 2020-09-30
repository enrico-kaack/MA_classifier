import os
import jinja2
import pandas as pd
import json
import argparse

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

def process_data(results):
    results["ratio_after_oversampling"][results.oversampling_enabled == False] = "-"
    results["ratio_after_undersampling"][results.undersampling_enabled == False] = "-"

    #important fix, check later if all experiments rerun
    results = results[results["max_vocab_size"] == 100000]
    results = results[results["input_src_path"] == "final_dataset"]

    merge_columns = ['window_size', 'step_size', 'encode_type','oversampling_enabled', 'ratio_after_oversampling', 'undersampling_enabled', 'ratio_after_undersampling', 'n_trees_in_forest', 'max_features', 'class_weight']

    random_forest_RN = results[(results["problem_type"] == "RETURN_NULL")  & (results["n_trees_in_forest"].notnull())]
    random_forest_RN.columns = random_forest_RN.columns.map(lambda a: a + "_RF" if a not in merge_columns else a)

    random_forest_CC = results[(results["problem_type"] == "RETURN_NULL")  & (results["n_trees_in_forest"].notnull())]
    random_forest_CC.columns = random_forest_CC.columns.map(lambda a: a + "_CC" if a not in merge_columns else a)

    random_forest_CCS = results[(results["problem_type"] == "RETURN_NULL")  & (results["n_trees_in_forest"].notnull())]
    random_forest_CCS.columns = random_forest_CCS.columns.map(lambda a: a + "_CCS" if a not in merge_columns else a)


    merged = random_forest_RN.merge(random_forest_CC, on=merge_columns, suffixes=("_LEFT1", "_RIGHT1")).merge(random_forest_CCS, on=merge_columns, suffixes=("_LEFT2", "_RIGHT2"))
    return merged

def read_data(input_dir):
    frames = []
    for root, dirs, files in os.walk(input_dir):
        for name in files:
            if name.endswith(".json"):
                frame = pd.read_json(os.path.join(root, name), orient="record", lines=True)
                frames.append(frame)
    results = pd.concat(frames)
    return results

def print_data(data, output_file, template_file):
    with open(template_file, "r") as f:
        t = latex_jinja_env.from_string(f.read())
        rendered = t.render(data=merged.T.to_dict().values())
        with open(output_file, "w") as o:
            o.write(rendered)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Write tex table for all train data')
    parser.add_argument('input', type=str, help='Input dir for dataset')
    parser.add_argument('template', type=str, help='File for tempalte')
    parser.add_argument('output', type=str, help='Output File path')
    args = parser.parse_args()

    results = read_data(args.input)
    merged = process_data(results)
    print_data(merged, args.output, args.template)