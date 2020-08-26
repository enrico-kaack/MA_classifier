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
    for type in problem_types:
        total_number_of_problem_type = 0
        for problem_file in parsed_files:
            for p in problem_file["problems"]:
                if p["type"] == type:
                    total_number_of_problem_type += 1

        #LOCs containing problems
        loc_containing_problems = total_number_of_problem_type / total_loc
        print(f"LOCs containing probem {type}: {loc_containing_problems}")

        #problems per file
        problems_per_file = total_number_of_problem_type / total_files
        print(f"Problems per file for {type}: {problems_per_file}")

