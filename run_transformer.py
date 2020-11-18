from tasks.preprocessing import ProblemType
from tasks.transformer import TaskEvaluateTransformer
import d6tflow


t = TaskEvaluateTransformer(problem_type=ProblemType.RETURN_NONE, input_src_path="final_dataset", oversampling_enabled=False, undersampling_enabled=True, 
    ratio_after_undersampling=0.5)

d6tflow.run(t)
