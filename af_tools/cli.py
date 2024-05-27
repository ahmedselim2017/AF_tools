import argparse
import sys
from pathlib import Path
import multiprocessing
from itertools import repeat

import matplotlib
from matplotlib.figure import Figure

from af_tools.afparser import AFParser
from af_tools.output_types import AFPrediction
from af_tools.afplotter import AFPlotter


def get_args() -> dict:
    parser = argparse.ArgumentParser()

    parser.add_argument("--af_dir",
                        help="Output directory of Alphafold",
                        required=True)
    parser.add_argument("--fig_dir",
                        help="Directory for figure outputs",
                        required=True)
    parser.add_argument(
        "-n",
        "--process_number",
        help="Number of processes to use while multiprocessing",
        default=1,
        type=int,
        required=False)
    parser.add_argument(
        "--rmsd_plddt",
        action="store_true",
        help="""Plot the RMSD vs pLDDT plot. The RMSD values are caluclated with
        respect to the model with the highest pLDDT value.""")
    parser.add_argument("--rmsd_ranks",
                        type=int,
                        nargs="+",
                        help="Ranks that should be used to calcualte RMSDs.",
                        default=[1])
    parser.add_argument(
        "--rmsd_hdbscan",
        action="store_true",
        help="Use HDBSCAN clustering while plotting the RMSD vs pLDDT plot.")
    parser.add_argument("--plddt",
                        action="store_true",
                        help="Plot pLDDT values for all predictions.")
    parser.add_argument("--pae",
                        action="store_true",
                        help="Plot PAE graphs for all predictions.")
    parser.add_argument(
        "--plddt_hist",
        action="store_true",
        help="Graph a histogram for the mean pLDDT values of each model.")

    parser.add_argument("--plot_all",
                        action="store_true",
                        help="Plot all possible graphs.")

    args = parser.parse_args()

    return {
        "af_dir": args.af_dir,
        "fig_dir": args.fig_dir,
        "process_number": args.process_number,
        "plot_rmsd_plddt": args.rmsd_plddt,
        "use_rmsd_hdbscan": args.rmsd_hdbscan,
        "rmsd_ranks": args.rmsd_ranks,
        "plot_plddt": args.plddt,
        "plot_pae": args.pae,
        "plot_plddt_hist": args.plddt_hist,
        "plot_all": args.plot_all
    }


def plot_pred(pred: AFPrediction, args_dict: dict, fig_path: Path,
              pred_ind: int) -> None:
    plotter = AFPlotter()

    if args_dict["plot_plddt"] or args_dict["plot_all"]:
        save_fig(plotter.plot_plddt(pred),
                 fig_path / f"{pred_ind}-{pred.name}_plddt")

    if args_dict["plot_pae"] or args_dict["plot_all"]:
        save_fig(plotter.plot_pae(pred),
                 fig_path / f"{pred_ind}-{pred.name}_pae")


def plot_pred_wrapper(args: tuple) -> None:
    pred, args_dict, fig_path, pred_ind = args
    plot_pred(pred=pred,
              args_dict=args_dict,
              fig_path=fig_path,
              pred_ind=pred_ind)


def save_fig(fig: Figure, path_wo_ext: Path) -> None:

    matplotlib.use('agg')
    fig.savefig(path_wo_ext.with_suffix(".png"), bbox_inches="tight")

    matplotlib.use('pdf')
    fig.savefig(path_wo_ext.with_suffix(".pdf"), bbox_inches="tight")


def cli() -> None:
    args_dict = get_args()

    fig_path = Path(args_dict["fig_dir"])

    if fig_path.is_file():
        emsg = f"ERROR!! Given figure directory is a file: {args_dict['fig_dir']}\n"
        sys.stderr.write(emsg)
        sys.exit(1)
    elif not fig_path.is_dir():
        wmsg = f"""WARNING: Given figure output directory {args_dict['fig_dir']} does not
        exist. Creating necessary directories."""
        print(wmsg)

        fig_path.mkdir(parents=True)

    afoutput = AFParser(
        path=args_dict["af_dir"],
        process_number=args_dict["process_number"]).get_output()

    if args_dict["plot_rmsd_plddt"] or args_dict["plot_all"]:
        rank_ind = [x - 1 for x in args_dict["rmsd_ranks"]]
        rmsds, plddts = afoutput.calculate_rmsds_plddts(rank_indeces=rank_ind)
        if args_dict["use_rmsd_hdbscan"] or args_dict["plot_all"]:
            hbscan = afoutput.get_rmsd_plddt_hbscan(rmsds=rmsds, plddts=plddts)
            fig = afoutput.plot_rmsd_plddt(rmsds=rmsds,
                                           plddts=plddts,
                                           hbscan=hbscan)
            save_fig(fig=fig, path_wo_ext=fig_path / "rmsd_plddt_hdbscan")
        if not args_dict["use_rmsd_hdbscan"] or args_dict["plot_all"]:
            fig = afoutput.plot_rmsd_plddt(rmsds=rmsds, plddts=plddts)
            save_fig(fig=fig, path_wo_ext=fig_path / "rmsd_plddt")

    if args_dict["plot_plddt_hist"] or args_dict["plot_all"]:
        fig = afoutput.plot_plddt_hist()
        save_fig(fig=fig, path_wo_ext=fig_path / "plddt_hist")

    with multiprocessing.Pool(processes=args_dict["process_number"]) as pool:
        for _ in pool.imap_unordered(
                plot_pred_wrapper,
            [(pred, args_dict, fig_path, pred_ind)
             for pred_ind, pred in enumerate(afoutput.predictions)]):
            pass
