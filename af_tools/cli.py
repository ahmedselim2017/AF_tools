import argparse
import sys
import pathlib


def get_args() -> tuple:
    parser = argparse.ArgumentParser()

    parser.add_argument("--af_dir",
                        help="Output directory of Alphafold",
                        required=True)
    parser.add_argument("--fig_dir",
                        help="Directory for figure outputs",
                        required=True)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    return args.af_dir, args.fig_dir, args.verbose


def cli() -> None:
    dir_af, dir_fig, is_verbose = get_args()

    path_af = pathlib.Path(dir_af)
    path_fig = pathlib.Path(dir_fig)

    if not path_af.is_dir():
        emsg = f"ERROR!! Given Alphafold output directory is not a valid directory: {dir_af}\n"
        sys.stderr.write(emsg)
        sys.exit(1)
    elif path_fig.is_file():
        emsg = f"ERROR!! Given figure output directory is an existing file: {dir_fig}\n"
        sys.stderr.write(emsg)
        sys.exit(1)
    elif not path_fig.is_dir():
        msg = f"WARNING: Given figure output directory {dir_fig} does not exist. Creating necessary directories."
        print(msg)

        path_fig.mkdir(parents=True)
