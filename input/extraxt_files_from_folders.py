import random
import os
import shutil

# script to extract a random subset of files from the input dir to the output dir.

def run_extraction(input_dir, output_dir, ratio):
    all_files = []
    for dir_path, dir_names, file_names in os.walk(input_dir):
                for f in file_names:
                    _, file_extension = os.path.splitext(f)
                    if file_extension == ".py":
                        file_path = os.path.join(dir_path, f)
                        all_files.append(file_path)
    print(len(all_files))


    sample = random.sample(all_files, k=int(ratio*len(all_files)))
    print(len(all_files), len(sample))

    for f in sample:
        target_path = os.path.join(output_dir, f)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        shutil.move(f, target_path)

    with open("moved_files.txt", "w") as outfile:
        outfile.writelines("\n".join(sample))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='extract_files.py', description="Extract a random subset of files from input dir to the output dir")
    parser.add_argument("-i", "--input_dir", help="Input folder with files to move FROM", metavar="INPUT_DIR")
    parser.add_argument("-o", "--output_dir", help="Output dir of extracted files", metavar="OUTPUT_DIR")
    parser.add_argument("-r", "--ratio", help="Ratio of how many files should be moved. In range  [0.0,1.0]", metavar="RATIO")

    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    ratio = args.ratio

    run_extraction(input_dir, output_dir, ratio)
