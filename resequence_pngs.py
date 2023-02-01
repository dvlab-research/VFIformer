import os
import shutil
import glob
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='Resequence video frame PNG files')
    parser.add_argument("--path", default="./images", type=str, help="Path to files to resequence")
    parser.add_argument("--new_name", default="pngsequence", type=str, help="New filename, default 'pngsequence'")
    parser.add_argument("--start", default=0, type=int, help="Starting index, default 0")
    parser.add_argument("--step", default=1, type=int, help="Index step, default is 1")
    parser.add_argument("--copy", dest="copy", default=False, action="store_true", help="Copy rather than rename files")
    parser.add_argument("--verbose", dest="verbose", default=False, action="store_true", help="Show extra details")
    args = parser.parse_args()

    init_log(args.verbose)

    files = glob.glob(os.path.join(args.path, "*.png"))

    files = sorted(os.listdir(args.path))
    num_files = len(files)
    log(f"Found {num_files} files")

    max_file_num = num_files * args.step
    num_width = len(str(max_file_num))

    index = 0
    for file in tqdm(files):
        new_filename = args.new_name + str(index).zfill(num_width) + ".png"
        old_filepath = os.path.join(args.path, file)
        new_filepath = os.path.join(args.path, new_filename)

        if args.copy:
            shutil.copy(old_filepath, new_filepath)
            log(f"File {file} copied to {new_filename}")
        else:
            os.replace(old_filepath, new_filepath)
            log(f"File {file} renamed to {new_filename}")

        index += args.step

global verbose
verbose = False

def init_log(verbose_enabled):
    global verbose
    verbose = verbose_enabled

def log(message):
    global verbose
    if verbose:
        print(message)

if __name__ == '__main__':
    main()
