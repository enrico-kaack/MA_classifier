from rules.return_none import ReturnNoneRule
from tqdm.autonotebook import tqdm
import logging

def run_rules_on_dataset(dataset):
    rules = [ReturnNoneRule()]
    processed_data = []
    print()
    for data in tqdm(dataset):
        problems = []
        for rule in rules:
            try:
                problems.extend( rule.analyse_source_code(data["src"]))
            except SyntaxError:
                logging.warn(f"Syntax error on file:{data['file_path']}. Ignoring file")
        processed_data.append({**data, "problems": problems})
    return processed_data    
