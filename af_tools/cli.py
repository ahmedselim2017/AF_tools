import argparse
import sys
from pathlib import Path
import multiprocessing
from itertools import repeat
import matplotlib


def get_args() -> tuple:
    parser = argparse.ArgumentParser()

    parser.add_argument("--af_dir",
                        help="Output directory of Alphafold",
                        required=True)
    parser.add_argument("--fig_dir",
                        help="Directory for figure outputs",
                        required=True)
    parser.add_argument("--rec",
                        action=argparse.BooleanOptionalAction,
                        help="Search the output directory recursively",
                        default=False,
                        required=False)
    parser.add_argument(
        "-n",
        "--process_number",
        help="Number of processes to use while multiprocessing",
        default=1,
        type=int,
        required=False)

    args = parser.parse_args()

    return args.af_dir, args.fig_dir, args.rec, args.process_number


def plot(pred: Prediction, path_fig: Path):
    afplotter = AFPlotter()
    fig_plddt = afplotter.plot_plddt(pred)
    fig_pae = afplotter.plot_pae(pred)

    matplotlib.use('agg')
    fig_plddt.savefig(path_fig / f"{pred.name}_plddt.png", bbox_inches="tight")
    fig_pae.savefig(path_fig / f"{pred.name}_pae.png", bbox_inches="tight")

    matplotlib.use('pdf')
    fig_plddt.savefig(path_fig / f"{pred.name}_plddt.pdf", bbox_inches="tight")
    fig_pae.savefig(path_fig / f"{pred.name}_pae.pdf", bbox_inches="tight")


def cli() -> None:
    dir_af, dir_fig, search_recursively, process_number = get_args()

    path_fig = Path(dir_fig)

    if path_fig.is_file():
        emsg = f"ERROR!! Given figure output directory is an existing file: {dir_fig}\n"
        sys.stderr.write(emsg)
        sys.exit(1)
    elif not path_fig.is_dir():
        msg = f"WARNING: Given figure output directory {dir_fig} does not exist. Creating necessary directories."
        print(msg)

        path_fig.mkdir(parents=True)

    afoutput = AFOutput(path=dir_af,
                        search_recursively=search_recursively,
                        process_number=process_number)

    with multiprocessing.Pool(processes=process_number) as pool:
        pool.starmap(plot, zip(afoutput.predictions, repeat(path_fig)))
