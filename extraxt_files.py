import random
import os
import shutil

input_dir = "final_dataset"
output_dir = "final_test"
ratio = 0.10


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