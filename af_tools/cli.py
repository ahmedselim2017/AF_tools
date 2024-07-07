import sys
from pathlib import Path
import multiprocessing
import pickle

import click
import matplotlib
from matplotlib.figure import Figure

from af_tools.afparser import AFParser
from af_tools.output_types import AFPrediction
from af_tools.afplotter import AFPlotter
from af_tools.data_types.afoutput import AFOutput
from af_tools.data_types.colabfold_msa import ColabfoldMSA


def plot_pred(pred: AFPrediction, what2plot, fig_path: Path,
              pred_ind: int) -> None:
    plotter = AFPlotter()
    for p in what2plot:
        match p:
            case "pred_plddt":
                save_fig(plotter.plot_plddt(pred),
                         fig_path / f"{pred_ind}-{pred.name}_plddt")
            case "pae":
                save_fig(plotter.plot_pae(pred),
                         fig_path / f"{pred_ind}-{pred.name}_pae")


def plot_pred_wrapper(args: tuple) -> None:
    pred, what2plot, fig_path, pred_ind = args
    plot_pred(pred=pred,
              what2plot=what2plot,
              fig_path=fig_path,
              pred_ind=pred_ind)


def save_fig(fig: Figure, path_wo_ext: Path) -> None:

    matplotlib.use('agg')
    fig.savefig(path_wo_ext.with_suffix(".png"), bbox_inches="tight")

    matplotlib.use('pdf')
    fig.savefig(path_wo_ext.with_suffix(".pdf"), bbox_inches="tight")


@click.command()
@click.option("--af_dir",
              help="AF output directory that will be analyzed",
              type=click.Path(exists=True,
                              file_okay=False,
                              readable=True,
                              resolve_path=True,
                              path_type=Path))
@click.option("--fig_dir",
              help="Output directory for figures",
              type=click.Path(file_okay=False,
                              writable=True,
                              resolve_path=True,
                              path_type=Path))
@click.option("-n",
              "--process_count",
              help="Process count",
              default=1,
              type=click.IntRange(min=1))
@click.option("--save_pickle",
              "pickle_save_path",
              help="Directory to save analysis results as a pickle",
              type=click.Path(file_okay=False,
                              writable=True,
                              resolve_path=True,
                              path_type=Path))
@click.option("--load_pickle",
              "pickle_load_path",
              help="Path of pickle that includes the analysis results.",
              type=click.Path(dir_okay=False,
                              readable=True,
                              resolve_path=True,
                              path_type=Path))
@click.option("--plot",
              "plot_types",
              help="Plot a selected type of graph",
              type=click.Choice(plots := [
                  "pred_plddt", "plddt_hist", "pae", "pairwise_RMSDs",
                  "ref_RMSDs", "ref_TMs", "pairwise_TMs"
              ],
                                case_sensitive=False),
              multiple=True)
@click.option("--plot_all",
              help="Plot all graphs",
              is_flag=True,
              default=False)
@click.option("--ref_structure",
              help=("Not yet implemented. Reference structure for RMSD"
                    "calculations. Uses the model"
                    "with th highest pLDDT by default"))
def analyze(af_dir: Path, fig_dir: Path | None, process_count: int,
            pickle_save_path: Path | None, pickle_load_path: Path | None,
            plot_types: tuple[str], plot_all: bool,
            ref_structure: str | None) -> None:
    if pickle_load_path is not None and af_dir is not None:
        emsg = "Error: --af_dir is mutually exclusive with --load_pickle."
        sys.stderr.write(emsg)
        sys.exit(1)
    elif pickle_load_path is None and af_dir is None:
        emsg = ("Error: Neither --af_dir or --load_pickle is given. There is"
                "no output to analyze.")
        sys.stderr.write(emsg)
        sys.exit(1)

    if fig_dir is None and (plot_all or plot_types is not None):
        emsg = "No figure directory is given. Can't plot the wanted plots."
        sys.stderr.write(emsg)
        sys.exit(1)

    afoutput: AFOutput | None = None
    if af_dir is not None:
        afoutput = AFParser(path=af_dir,
                            process_number=process_count,
                            sort_plddt=False).get_output()
    elif pickle_load_path is not None:
        with open(pickle_load_path, "rb") as pickle_load_fh:
            afoutput = pickle.load(pickle_load_fh)
    assert afoutput is not None

    what2plot = plots if plot_all else plot_types

    if fig_dir is not None:
        fig_dir.mkdir(parents=True, exist_ok=True)
        plotter = AFPlotter()

        if "plddt_hist" in what2plot:
            fig = afoutput.plot_plddt_hist()
            save_fig(fig=fig, path_wo_ext=fig_dir / "plddt_hist")
        if "ref_RMSDs" in what2plot:
            afoutput.ref_rmsds = afoutput.calculate_ref_rmsds(rank_index=0)
            fig = afoutput.plot_ref_rmsd_plddt()
            save_fig(fig=fig, path_wo_ext=fig_dir / "ref_rmsds")

            afoutput.ref_rmsds = afoutput.calculate_ref_rmsds(rank_index=0,
                                                              mult_conf=True)
            fig = afoutput.plot_data_conf(afoutput.ref_rmsds,
                                          datalabel="RMSD",
                                          rank_index=0,
                                          hdbscan=False)
            save_fig(fig=fig, path_wo_ext=fig_dir / "ref_rmsds_multimer_conf")
        if "pairwise_RMSDs" in what2plot:
            if afoutput.pairwise_rmsds is None:
                afoutput.pairwise_rmsds = afoutput.calculate_pairwise_rmsds(
                    rank_index=0)
            labels = [pred.name for pred in afoutput.predictions]
            fig = plotter.plot_upper_trig(afoutput.pairwise_rmsds)
            fig_log = plotter.plot_upper_trig(afoutput.pairwise_rmsds,
                                              labels=labels,
                                              log_scale=True)
            save_fig(fig=fig, path_wo_ext=fig_dir / "pairwise_rmsds")
            save_fig(fig=fig_log, path_wo_ext=fig_dir / "pairwise_rmsds_log")
        if "ref_TMs" in what2plot:
            fig = afoutput.plot_ref_tm_plddt()
            save_fig(fig=fig, path_wo_ext=fig_dir / "ref_tms")

            afoutput.ref_tms = afoutput.calculate_ref_tms(rank_index=0,
                                                          mult_conf=True)
            fig = afoutput.plot_data_conf(afoutput.ref_tms,
                                          datalabel="TM",
                                          rank_index=0,
                                          hdbscan=False)
            save_fig(fig=fig, path_wo_ext=fig_dir / "ref_tms_multimer_conf")
        if "pairwise_TMs" in what2plot:
            if afoutput.pairwise_tms is None:
                afoutput.pairwise_tms = afoutput.calculate_pairwise_tms(
                    rank_index=0)
            labels = [pred.name for pred in afoutput.predictions]
            fig = plotter.plot_upper_trig(afoutput.pairwise_tms, labels=labels)
            fig_log = plotter.plot_upper_trig(afoutput.pairwise_tms,
                                              log_scale=True,
                                              labels=labels)

            save_fig(fig=fig, path_wo_ext=fig_dir / "pairwise_tms")
            save_fig(fig=fig_log, path_wo_ext=fig_dir / "pairwise_tms_log")

    with multiprocessing.Pool(processes=process_count) as pool:
        for _ in pool.imap_unordered(
                plot_pred_wrapper,
            [(pred, what2plot, fig_dir, pred_ind)
             for pred_ind, pred in enumerate(afoutput.predictions)]):
            pass

    if pickle_save_path is not None:
        with open(pickle_save_path / "af_tools_afoutput.pickle",
                  "wb+") as pickle_save_fh:
            pickle.dump(afoutput, pickle_save_fh)


@click.command()
@click.option("--msa_path",
              help="Path of the MSA file.",
              type=click.Path(exists=True,
                              dir_okay=False,
                              readable=True,
                              resolve_path=True,
                              path_type=Path),
              required=True)
@click.option("--out_dir",
              help="Directory for subsampled msa files.",
              type=click.Path(file_okay=False,
                              readable=True,
                              resolve_path=True,
                              path_type=Path))
@click.option("--msa_length_2",
              help="Set the maximum MSA length to 2**[msa_length_2]",
              type=click.IntRange(min=1),
              default=9)
@click.option("--replicate_count",
              help="Number of generated MSAs with the same length.",
              type=click.IntRange(min=1),
              default=5)
def subsample_msa(msa_path, out_dir: Path | None, msa_length_2: int,
                  replicate_count: int) -> None:
    if msa_path.suffix != ".a3m":
        emsg = (f"Given MSA file {msa_path.name} has a different filetype than"
                "a3m. Only a3m files are supported.")
        sys.stderr.write(emsg)
        sys.exit(1)

    if out_dir is None:
        out_dir = Path(msa_path.parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    msa = ColabfoldMSA(msa_path)
    for i in range(msa_length_2 + 1):
        msa.sample_records(sample_lenght=2**i,
                           sample_count=replicate_count,
                           output_dir=out_dir,
                           save_samples=True)


@click.group(context_settings={'show_default': True})
def cli() -> None:
    pass


cli.add_command(analyze)
cli.add_command(subsample_msa)
