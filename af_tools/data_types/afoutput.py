import multiprocessing as mp
from pathlib import Path
from itertools import batched
import pickle
import subprocess
from collections.abc import Sequence

from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray

from sklearn.cluster import HDBSCAN
from tqdm import tqdm

from af_tools.afplotter import AFPlotter
from af_tools.output_types import AFModel
from af_tools import utils


class AFOutput:

    def __init__(self,
                 path: Path,
                 ref_path: Path | None = None,
                 process_number: int | None = None,
                 should_load: list[str] | None = None):
        self.path = self.check_path(path)
        self.ref_path = ref_path
        self.should_load = should_load if should_load else set(
            ["mean_plddt", "mult_conf", "ptm", "iptm"])

        self.predictions = self.get_predictions()

        self.ref_rmsds: NDArray | None = None
        self.pairwise_rmsds: NDArray | None = None
        self.ref_tms: NDArray | None = None
        self.pairwise_tms: NDArray | None = None

    def check_path(self, path: str | Path) -> Path:
        if isinstance(path, str):
            p = Path(path)
        else:
            p = path
        if not p.is_dir():
            raise Exception(
                f"Alphafold output directory is not a valid directory: {p}")
        return p

    def get_predictions(self) -> Sequence:
        raise Exception("Can't get predictions")

    def plot_all_plddts(self) -> list[Figure]:
        figures: list[Figure] = []
        plotter = AFPlotter()
        for pred in self.predictions:
            figures.append(plotter.plot_plddt(pred))
        return figures

    def plot_all_paes(self) -> list[Figure]:
        figures: list[Figure] = []
        plotter = AFPlotter()
        for pred in self.predictions:
            figures.append(plotter.plot_pae(pred))
        return figures

    def plot_plddt_hist(self,
                        use_color: bool = True,
                        draw_mean: bool = True) -> Figure:
        plotter = AFPlotter()

        predicted_models: list[AFModel] = []
        for pred in self.predictions:
            predicted_models += pred.models

        fig = plotter.plot_plddt_hist(predicted_models=predicted_models,
                                      use_color=use_color,
                                      draw_mean=draw_mean)

        return fig

    def get_rank_paths(self, rank_index: int) -> list[Path]:
        model_paths: list[Path] = []
        for pred in self.predictions:
            model = pred.models[rank_index]
            model_paths.append(model.get_best_model_path())
        return model_paths

    def get_ref_path(self, rank_index: int, mult_conf: bool = False) -> Path:
        ref_path: Path | None = None
        if self.ref_path is None:
            if mult_conf:
                ref_model = max(
                    [pred.models[rank_index] for pred in self.predictions],
                    key=lambda x: x.multimer_conf)
            else:
                ref_model = max(
                    [pred.models[rank_index] for pred in self.predictions],
                    key=lambda x: x.mean_plddt)
            ref_path = ref_model.get_best_model_path()
        else:
            ref_path = self.ref_path
        assert ref_path is not None
        return ref_path

    def calculate_pairwise_rmsds(self,
                                 rank_index: int,
                                 mult_conf=False) -> NDArray:
        rmsds = np.full((len(self.predictions), len(self.predictions)),
                        np.nan,
                        dtype=float)
        np.fill_diagonal(rmsds, 0)
        model_paths = self.get_rank_paths(rank_index)

        if self.process_number > 1:
            with mp.Pool(processes=self.process_number) as pool:
                results = pool.starmap(
                    utils.calculate_rmsd,
                    tqdm([(model_paths[i + 1:], m)
                          for i, m in enumerate(model_paths[:-1])],
                         total=len(model_paths),
                         desc="Calculating pairwise RMSDs"))

                for i, result in enumerate(results):
                    rmsds[i, i + 1:] = result
        else:
            pbar_mpaths = tqdm(model_paths[:-1])
            for i, m in enumerate(pbar_mpaths):
                pbar_mpaths.desc = f"Calculating pairwise RMSDs of{m.name}"
                rmsds[i, i + 1:] = utils.calculate_rmsd(model_paths[i + 1:], m)

        return rmsds

    def calculate_ref_rmsds(self, rank_index: int, mult_conf=False) -> NDArray:
        ref_rmsds = np.full(len(self.predictions), np.nan, dtype=float)
        model_paths = self.get_rank_paths(rank_index)

        ref_structure = utils.load_structure(
            self.get_ref_path(rank_index, mult_conf))

        if self.process_number > 1:
            with mp.Pool(processes=self.process_number) as pool:
                results = pool.starmap(
                    utils.calculate_rmsd,
                    tqdm([(m, ref_structure) for m in model_paths],
                         total=len(model_paths),
                         desc="Calculating reference RMSDs"))
                for i, r in enumerate(results):
                    ref_rmsds[i] = r

        else:
            pbar_mpaths = tqdm(model_paths)
            for i, m in enumerate(pbar_mpaths):
                pbar_mpaths.desc = f"Calculating reference RMSDs of {m.name}"
                ref_rmsds[i] = utils.calculate_rmsd(m, ref_structure)

        return ref_rmsds

    def calculate_pairwise_tms(self, rank_index: int) -> NDArray:
        from shutil import which
        if which("USalign") is None:
            raise FileNotFoundError(
                "USalign executable can't be found on PATH.")

        tms = np.full((len(self.predictions), len(self.predictions)),
                      np.nan,
                      dtype=float)
        np.fill_diagonal(tms, 1)
        model_paths = self.get_rank_paths(rank_index)

        if self.process_number > 1:
            with mp.Pool(processes=self.process_number) as pool:
                results = pool.starmap(
                    utils.calculate_tm,
                    tqdm([(model_paths[i + 1:], m)
                          for i, m in enumerate(model_paths[:-1])],
                         total=len(model_paths),
                         desc="Calculating pairwise TMs"))

                for i, result in enumerate(results):
                    tms[i, i + 1:] = result
        else:
            pbar_mpaths = tqdm(model_paths[:-1])
            for i, m in enumerate(pbar_mpaths):
                pbar_mpaths.desc = f"Calculating pairwise TMs of{m.name}"
                tms[i, i + 1:] = utils.calculate_tm(model_paths[i + 1:], m)

        return tms

    def calculate_ref_tms(self, rank_index: int, mult_conf=False) -> NDArray:
        ref_tms = np.full(len(self.predictions), np.nan, dtype=float)
        model_paths = self.get_rank_paths(rank_index)

        ref_path = self.get_ref_path(rank_index, mult_conf)

        if self.process_number > 1:
            with mp.Pool(processes=self.process_number) as pool:
                results = pool.starmap(
                    utils.calculate_tm,
                    tqdm([(m, ref_path) for m in model_paths],
                         total=len(model_paths),
                         desc="Calculating reference TMs"))
                for i, r in enumerate(results):
                    ref_tms[i] = r
        else:
            pbar_mpaths = tqdm(model_paths)
            for i, m in enumerate(pbar_mpaths):
                pbar_mpaths.desc = f"Calculating reference TMs of {m.name}"
                ref_tms[i] = utils.calculate_rmsd(m, ref_path)

        return ref_tms

    def plot_ref_rmsd_plddt(self,
                            rank_index: int = 0,
                            hdbscan: bool = True) -> Figure:
        if self.ref_rmsds is None:
            ref_rmsds = self.calculate_ref_rmsds(rank_index)
        else:
            ref_rmsds = self.ref_rmsds
        return self.plot_data_plddt(ref_rmsds,
                                    datalabel="RMSD",
                                    rank_index=rank_index,
                                    hdbscan=hdbscan)

    def plot_ref_tm_plddt(self,
                          rank_index: int = 0,
                          hdbscan: bool = True) -> Figure:
        if self.ref_tms is None:
            ref_tms = self.calculate_ref_tms(rank_index)
        else:
            ref_tms = self.ref_tms

        return self.plot_data_plddt(ref_tms,
                                    datalabel="TM Score",
                                    rank_index=rank_index,
                                    hdbscan=hdbscan)

    def plot_data_plddt(self,
                        data: NDArray,
                        datalabel: str = "Data",
                        rank_index: int = 0,
                        hdbscan: HDBSCAN | bool | None = None) -> Figure:
        mean_plddts = np.full(len(self.predictions), np.nan, dtype=float)
        for i, pred in enumerate(self.predictions):
            mean_plddts[i] = pred.models[rank_index].mean_plddt

        labels = None
        if hdbscan is True:
            hdbscan = self.get_plddt_hdbscan(data, mean_plddts)
            labels = hdbscan.labels_
        elif isinstance(hdbscan, HDBSCAN):
            labels = hdbscan.labels_

        plotter = AFPlotter()
        return plotter.plot_data_plddt(data,
                                       mean_plddts,
                                       datalabel,
                                       labels=labels)

    def plot_data_conf(self,
                       data: NDArray,
                       datalabel: str = "Data",
                       rank_index: int = 0,
                       hdbscan: HDBSCAN | bool | None = None) -> Figure:
        confs = np.full(len(self.predictions), np.nan, dtype=float)
        for i, pred in enumerate(self.predictions):
            confs[i] = pred.models[rank_index].multimer_conf

        labels = None
        if hdbscan is True:
            hdbscan = self.get_plddt_hdbscan(data, confs)
            labels = hdbscan.labels_
        elif isinstance(hdbscan, HDBSCAN):
            labels = hdbscan.labels_

        plotter = AFPlotter()
        return plotter.plot_data_conf(data, confs, datalabel, labels=labels)

    def get_plddt_hdbscan(self,
                          data: NDArray,
                          plddts: NDArray,
                          min_sample_size: int = 2) -> HDBSCAN:
        from sklearn.cluster import HDBSCAN
        hdb = HDBSCAN(min_samples=min_sample_size, n_jobs=self.process_number)
        hdb.fit(np.dstack((data, plddts))[0])
        return hdb

    def get_rmsd_plddt_cluster_paths(self, rank_index: int,
                                     hdbscan: HDBSCAN) -> tuple:
        model_paths = self.get_rank_paths(rank_index)
        mean_plddts = np.full(len(self.predictions), np.nan, dtype=float)
        for i, pred in enumerate(self.predictions):
            mean_plddts[i] = pred.models[rank_index].mean_plddt

        clusters = np.unique(hdbscan.labels_)
        clusters = clusters[clusters != 1]

        cluster_paths: list[list[Path]] = []
        cluster_plddts = np.full(len(clusters), np.nan, dtype=float)
        for i, cluster in enumerate(clusters):
            selected_indices = np.where(hdbscan.labels_ == cluster)
            cluster_plddts[i] = np.mean(mean_plddts[selected_indices])
            cluster_paths.append([model_paths[s] for s in selected_indices[0]])

        return cluster_paths, cluster_plddts
