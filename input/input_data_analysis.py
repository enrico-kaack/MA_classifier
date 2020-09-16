from tasks.preprocessing import ProblemType


def analyse_parsed_data(parsed_files):
    total_loc = 0
    for f in parsed_files:
        total_loc += len(f["src"].splitlines())
    print(f"Lines of Code: {total_loc}")

    total_files = len(parsed_files)
    print(f"Total Files: {total_files}")

    print(f"LOCs per file {total_loc / total_files}")

    problem_types = [ProblemType.RETURN_NONE.value, ProblemType.CONDITION_COMPARISON.value]
    for t in problem_types:
        total_number_of_problem_type = 0
        for problem_file in parsed_files:
            for p in problem_file["problems"]:
                if p["type"] == t:
                    total_number_of_problem_type += 1

        #LOCs containing problems
        loc_containing_problems = total_number_of_problem_type / total_loc
        print(f"LOCs containing probem {t}: {loc_containing_problems}")

        #problems per file
        problems_per_file = total_number_of_problem_type / total_files
        print(f"Problems per file for {t}: {problems_per_file}")

def analyse_problems_per_project(parsed_files, project_name_depth):
    results = {}
    problem_types = [ProblemType.RETURN_NONE.value, ProblemType.CONDITION_COMPARISON.value]
    for problem_file in parsed_files:
        file_path = problem_file["file_path"]
        key = "/".join(file_path.split("/")[:project_name_depth])
        file_info = results.get(key, {})
        for p in problem_file["problems"]:
            count = file_info.get(p["type"], 0)
            file_info[p["type"]] = count + 1
        results[key] = file_info
    for key, value in results.items():
        print(f" {key}: {value}")
    return results

def analyse_files_per_project(parsed_files, project_name_depth):
    results = {}
    for problem_file in parsed_files:
        file_path = problem_file["file_path"]
        key = "/".join(file_path.split("/")[:project_name_depth])
        numb = results.get(key, 0)
        results[key] = numb + 1

    for key, value in results.items():
        print(f" {key}: {value}, {value / len(parsed_files)}")
    return results



