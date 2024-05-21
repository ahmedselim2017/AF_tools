import argparse
import sys
import pathlib


from .analyze import AFOutput, AFPlotter


def get_args() -> tuple:
    parser = argparse.ArgumentParser()

    parser.add_argument("--af_dir", help="Output directory of Alphafold", required=True)
    parser.add_argument("--fig_dir", help="Directory for figure outputs", required=True)
    parser.add_argument("--rec", action=argparse.BooleanOptionalAction, help="Search the output directory recursively", default=False, required=False)

    args = parser.parse_args()

    return args.af_dir, args.fig_dir, args.rec


def cli() -> None:
    dir_af, dir_fig, search_recursively = get_args()

    path_fig = pathlib.Path(dir_fig)

    if path_fig.is_file():
        emsg = f"ERROR!! Given figure output directory is an existing file: {dir_fig}\n"
        sys.stderr.write(emsg)
        sys.exit(1)
    elif not path_fig.is_dir():
        msg = f"WARNING: Given figure output directory {dir_fig} does not exist. Creating necessary directories."
        print(msg)

        path_fig.mkdir(parents=True)

    afoutput = AFOutput(dir_af, search_recursively)
    afplotter = AFPlotter()

    for pred in afoutput.predictions:
        continue
        fig_plddt = afplotter.plot_plddt(pred)
        fig_plddt.savefig(path_fig / f"{pred.name}_plddt.png", bbox_inches="tight")
        fig_plddt.savefig(path_fig / f"{pred.name}_plddt.pdf", bbox_inches="tight")

        fig_pae = afplotter.plot_pae(pred)
        fig_pae.savefig(path_fig / f"{pred.name}_pae.png", bbox_inches="tight")
        fig_pae.savefig(path_fig / f"{pred.name}_pae.pdf", bbox_inches="tight")
