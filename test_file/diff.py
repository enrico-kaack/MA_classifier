import os
import hashlib

def sha256sum(filename):
    h  = hashlib.sha256()
    b  = bytearray(128*1024)
    mv = memoryview(b)
    with open(filename, 'rb', buffering=0) as f:
        for n in iter(lambda : f.readinto(mv), 0):
            h.update(mv[:n])
    return h.hexdigest()


first_files= {}
for dir_path, dir_names, file_names in os.walk("test-dir"):
            for f in file_names:
                file_name, file_extension = os.path.splitext(f)
                if file_extension == ".py":
                    sha256 = sha256sum(os.path.join(dir_path, f))
                    same_files = first_files.get(sha256, [])
                    same_files.append(os.path.join(dir_path, f))
                    first_files[sha256] = same_files
for dir_path, dir_names, file_names in os.walk("second_large_dataset"):
            for f in file_names:
                file_name, file_extension = os.path.splitext(f)
                if file_extension == ".py":
                    sha256 = sha256sum(os.path.join(dir_path, f))
                    same_files = first_files.get(sha256, [])
                    same_files.append(os.path.join(dir_path, f))
                    first_files[sha256] = same_files

len_same = 0
len_uneuqal = 0
for key, value in first_files.items():
    if len(value) == 2:
        len_same += 1
    else:
        len_uneuqal += 1
print(len_same, len_uneuqal)