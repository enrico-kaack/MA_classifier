import json
from json import JSONEncoder
import numpy

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def dump_json(task_id, parameters, results):
    results = {"task_id": task_id, **parameters, **results}
    with open(f"results/{task_id}.json", "w") as f:
            value = json.dumps(results, cls=NumpyArrayEncoder)
            f.write(value)