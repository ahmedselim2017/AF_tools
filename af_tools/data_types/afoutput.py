import multiprocessing as mp
from pathlib import Path
from itertools import batched
import pickle
import subprocess
from typing import Sequence

from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray

from Bio.PDB.Structure import Structure
from sklearn.cluster import HDBSCAN
from tqdm import tqdm

from af_tools.afplotter import AFPlotter
from af_tools.output_types import AFModel
from af_tools import utils


class AFOutput:

    def __init__(self,
                 path: Path,
                 ref_path: Path | None = None,
                 process_number: int = 1,
                 search_recursively: bool = False,
                 sort_plddt: bool = True):
        self.path = self.check_path(path)
        self.ref_path = ref_path
        self.process_number = process_number
        self.search_recursively = search_recursively
        self.sort_plddt = sort_plddt

        self.predictions = self.get_predictions()

        self.rmsds: NDArray | None = None
        self.tms: NDArray | None = None
        self.pickle_path: Path | None = None

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
            if hasattr(model, "relaxed_pdb_path"):
                model_paths.append(model.relaxed_pdb_path)
            else:
                model_paths.append(model.model_path)
        return model_paths

    def calculate_pairwise_rmsds(self, rank_index: int) -> NDArray:
        rmsds = np.full((len(self.predictions), len(self.predictions)),
                        np.nan,
                        dtype=float)
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

    def calculate_ref_rmsds(self, rank_index: int):
        ref_rmsds = np.full(len(self.predictions), np.nan, dtype=float)
        model_paths = self.get_rank_paths(rank_index)

        if self.ref_path is None:
            raise TypeError("No path for reference structure is given")

        ref_structure = utils.load_structure(self.ref_path)

        if self.process_number > 1:
            with mp.Pool(processes=self.process_number) as pool:
                ref_rmsds = pool.starmap(
                    utils.calculate_rmsd,
                    tqdm([(m, ref_structure) for m in model_paths],
                         total=len(model_paths),
                         desc="Calculating reference RMSDs"))
        else:
            pbar_mpaths = tqdm(model_paths)
            for i, m in enumerate(pbar_mpaths):
                pbar_mpaths.desc = f"Calculating reference RMSDs of {m.name}"
                ref_rmsds[i] = utils.calculate_rmsd(m, ref_structure)

        return ref_rmsds

    def calculate_all_vs_all_tms(self, rank_index: int = 0) -> NDArray:
        from shutil import which
        if which("USalign") is None:
            raise Exception(
                "ERROR: USalign can't be found. Can't calcualte TM values.")

        tms = np.full((len(self.predictions), len(self.predictions)),
                      np.nan,
                      dtype=float)
        model_paths = np.empty(len(self.predictions), dtype=np.dtypes.StrDType)

        for i, pred in enumerate(self.predictions):
            model = pred.models[rank_index]
            if hasattr(model, "relaxed_pdb_path"):
                model_paths[i] = str(model.relaxed_pdb_path)
            else:
                model_paths[i] = str(model.model_path)

        # TODO multiprocessing?

        for i, m1 in enumerate(tqdm(model_paths)):
            for j, m2 in enumerate(tqdm(model_paths, leave=False)):
                if i < j:
                    tms[i][j] = 1
                p = subprocess.run(["USalign", "-outfmt", "2", m1, m2],
                                   capture_output=True,
                                   text=True)
                tms[i][j] = float(p.stdout.split("\n")[1].split()[3])
        return tms

    def calculate_rmsds_plddts(
        self,
        rank_indeces: list[int] | range | None = None,
        ref_index: int | None = None,
        ref_structure: Structure | None = None,
    ) -> tuple[NDArray, NDArray]:

        if rank_indeces is None:
            rank_indeces = range(len(self.predictions[0].models))

        model_paths = np.empty(len(self.predictions), dtype=np.dtypes.StrDType)
        plddts = np.full(len(self.predictions) * len(rank_indeces),
                         np.nan,
                         dtype=float)

        for i, pred in enumerate(self.predictions):
            for j, rank_index in enumerate(rank_indeces):
                model = pred.models[rank_index]
                plddts[i * len(rank_indeces) + j] = model.mean_plddt
                if hasattr(model, "relaxed_pdb_path"):
                    model_paths[i * len(rank_indeces) +
                                j] = model.relaxed_pdb_path
                else:
                    model_paths[i * len(rank_indeces) + j] = model.model_path

        if ref_structure is None and ref_index is None:
            # TODO: isnt the plddts are already sorted?
            max_plddt_ind = np.argmax(plddts)
            ref_structure = utils.load_structure(model_paths[max_plddt_ind])
        elif ref_index:
            ref_structure = utils.load_structure(model_paths[ref_index])

        assert ref_structure

        rmsds = np.full(len(model_paths), np.nan, dtype=float)

        if self.process_number > 1:
            with mp.Pool(processes=self.process_number) as pool:
                results = tqdm(pool.imap_unordered(
                    utils.worker_wrapper_calculate_rmsd,
                    [(ref_structure, m_path, i)
                     for i, m_path in enumerate(model_paths)]),
                               desc="Calculating RMSDs",
                               total=len(model_paths))

                for result in results:
                    rmsds[result[0]] = result[1]
        else:
            pbar = tqdm(model_paths)
            for i, structure in enumerate(pbar):
                rmsds[i] = utils.worker_calculate_rmsd(ref_structure,
                                                       structure, i)[1]
                pbar.set_description(f"Calculating RMSDs of {structure}")

        return rmsds, plddts

    def plot_rmsd_plddt(self,
                        rmsds: NDArray,
                        plddts: NDArray,
                        hbscan: HDBSCAN | None = None) -> Figure:

        labels: NDArray | None = None
        if hbscan is not None:
            labels = hbscan.labels_

        plotter = AFPlotter()
        return plotter.plot_rmsd_plddt(plddts, rmsds, labels=labels)

    def get_rmsd_plddt_hbscan(self,
                              rmsds: NDArray,
                              plddts: NDArray,
                              min_sample_size: int = 2) -> HDBSCAN:
        from sklearn.cluster import HDBSCAN
        hdb = HDBSCAN(min_samples=min_sample_size)
        hdb.fit(np.dstack((rmsds, plddts))[0])
        return hdb

    def get_rmsd_plddt_cluster_paths(self, rank_indeces: list[int] | range,
                                     hbscan: HDBSCAN) -> tuple:
        model_paths = np.empty(len(self.predictions) * len(rank_indeces),
                               dtype=np.dtypes.StrDType)
        plddts = np.full(len(self.predictions) * len(rank_indeces),
                         np.nan,
                         dtype=float)

        for i, pred in enumerate(self.predictions):
            for j, rank_index in enumerate(rank_indeces):
                model = pred.models[rank_index]
                plddts[i * len(rank_indeces) + j] = model.mean_plddt
                if hasattr(model, "relaxed_pdb_path"):
                    model_paths[i * len(rank_indeces) + j] = str(
                        model.relaxed_pdb_path)
                else:
                    model_paths[i * len(rank_indeces) + j] = str(
                        model.model_path)

        clusters = np.unique(hbscan.labels_)
        clusters = clusters[clusters != 1]

        cluster_paths: list[NDArray] = []
        cluster_plddts = np.full(len(clusters), np.nan, dtype=float)
        for i, cluster in enumerate(clusters):
            selected_indices = np.where(hbscan.labels_ == cluster)
            cluster_plddts[i] = np.mean(plddts[selected_indices])
            cluster_paths.append(np.unique(model_paths[selected_indices]))

        return cluster_paths, cluster_plddts
