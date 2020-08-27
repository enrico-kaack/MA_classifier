from rules.return_none import ReturnNoneRule
from rules.condition_with_comparison import ConditionComparison
from rules.condition_comparison_simple import ConditionComparisonSimple
from tqdm.autonotebook import tqdm
import logging

def run_rules_on_dataset(dataset):
    rules = [ReturnNoneRule(), ConditionComparison(), ConditionComparisonSimple()]
    processed_data = []
    for data in tqdm(dataset):
        problems = []
        for rule in rules:
            try:
                rule_problems = rule.analyse_source_code(data["src"])
                problems.extend(rule_problems)
            except SyntaxError:
                logging.warn(f"Syntax error on file:{data['file_path']}. Ignoring file")
        processed_data.append({**data, "problems": problems})
    return processed_data    
