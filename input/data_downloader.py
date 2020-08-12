import logging
from zipfile import ZipFile
import os
import wget
from tqdm.autonotebook import tqdm
import argparse
import sys

class DataDownloader():
    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = output_dir
        self.count_python_files = 0

    def download_repos_in_org_repo_list(self, repos):
        logging.info(f"Downloading {len(repos)} repositories...")
        for entry in tqdm(repos):
            org, repo, branch = self._split_file_line(entry)

            logging.debug(f"Downloading {org}/{repo}/{branch}")
            zip_file = self.download_repository(org, repo, branch)
            extracted_path = self.extract_repository(zip_file, org, repo, branch)
            os.remove(zip_file)

        logging.info("Removing non-python files")
        self.remove_non_python_file()
        logging.info(f"Total number of python files: {self.count_python_files}")


    def _split_file_line(self, line):
        line = line.strip()
        splitted = line.split("/")
        return tuple(splitted)

    def url_constructor(self, org, repo, branch):
        return f"https://github.com/{org}/{repo}/archive/{branch}.zip"

    def download_repository(self, org, repo, branch):
        git_url = self.url_constructor(org, repo, branch)
        filename = wget.download(git_url, bar=None)
        return filename

    def extract_repository(self, file_name, org, repo, branch):
        target_path = os.path.join(self.output_dir, f"{org}/{repo}/{branch}")

        if os.path.exists(target_path):
            logging.error(f"Target path {target_path} already exists")
            return
        os.makedirs(target_path)
        with ZipFile(file_name, "r") as zipObject:
            extracted_name = zipObject.extractall(target_path)
            return target_path

    def remove_non_python_file(self):
        for dir_path, dir_names, file_names in os.walk(self.output_dir):
            for f in file_names:
                _, file_extension = os.path.splitext(f)
                if file_extension != ".py":
                    os.remove(os.path.join(dir_path, f))
                else:
                    self.count_python_files += 1

    def download_repos_in_file(self, file_name):
        logging.info(f"Read from {file_name}")
        with open(file_name) as input_file:
            lines = input_file.readlines()
            logging.debug(f"Read {len(lines)} from {file_name}")
            self.download_repos_in_org_repo_list(lines)

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='data_downlaoder.py', description="Downlaods repositories from github and removes non-python files")
    parser.add_argument("repo_list", help="File with a list of repos to download", metavar="REPO_LIST")
    parser.add_argument("-o", "--output_dir", help="File with a list of repos to download", metavar="OUTPUT_DIR")
    parser.add_argument("-l", "--log_level", help="Log level (DEBUG, INFO, ERROR)", metavar="LOG_LEVEL")

    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    repo_list_file = args.repo_list
    output_dir = args.output_dir

    if args.log_level != None:
        logging.basicConfig(level=getattr(logging, args.log_level, logging.INFO))
        

    downloader = DataDownloader(output_dir)
    downloader.download_repos_in_file(repo_list_file)

