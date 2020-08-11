import os
import pickle
import argparse
import sys

class AllFileReader():
    def __init__(self):
        super().__init__()
        self.data = []

    
    def load_source_file_from(self, dir):
        for dir_path, dir_names, file_names in os.walk(dir):
            for f in file_names:
                _, file_extension = os.path.splitext(f)
                if file_extension == ".py":
                    file_path = os.path.join(dir_path, f)
                    with open(file_path, "r") as open_file:
                        file_content = open_file.read()
                        self.data.append({
                            "file_path": file_path,
                            "src": file_content
                        })

    def dump_to_file(self, out_file):
        with open(out_file, "wb") as out:
            pickle.dump(self.data, out)
    
    def parse_from_pickle_file(self, in_file):
        with open(in_file, "rb") as f:
            self.data = pickle.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='read_all_to_json.py', description="Parses all source files into a compact pickle representation")
    parser.add_argument("input_dir", help="Input directory containing all source files", metavar="INPUT_DIR")
    parser.add_argument("output_file", help="Output pickle .pkl file", metavar="OUTPUT")
    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    reader = AllFileReader()
    reader.load_source_file_from(args.input_dir)
    reader.dump_to_file(args.output_file)