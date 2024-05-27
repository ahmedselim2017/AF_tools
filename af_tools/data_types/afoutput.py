import multiprocessing as mp
from pathlib import Path
from typing import Sequence

from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray

from Bio.PDB.Structure import Structure
from sklearn.cluster import HDBSCAN

from af_tools.afplotter import AFPlotter
from af_tools.output_types import AFModel
from af_tools import utils


class AFOutput:

    def __init__(self, path: str | Path, process_number: int = 1):
        self.path = self.check_path(path)
        self.process_number = process_number
        self.predictions = self.get_predictions()
        self.rmsds: NDArray | None = None

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

    def calculate_rmsds_plddts(
        self,
        rank_indeces: list[int] | range,
        ref_index: int | None = None,
        ref_structure: Structure | None = None,
    ) -> tuple[NDArray, NDArray]:

        model_paths: list[Path] = []  # NOTE: Numpy string array for paths?
        plddts = np.full(len(self.predictions) * len(rank_indeces), 0.0)

        for i, pred in enumerate(self.predictions):
            for j, rank_index in enumerate(rank_indeces):
                model = pred.models[rank_index]
                plddts[i * len(rank_indeces) + j] = model.mean_plddt
                model_paths.append(model.relaxed_pdb_path if hasattr(
                    model, "relaxed_pdb_path") else model.model_path)

        if ref_structure is None and ref_index is None:
            max_plddt_ind = np.argmax(plddts)
            ref_structure = utils.load_structure(model_paths[max_plddt_ind])
        elif ref_index:
            ref_structure = utils.load_structure(model_paths[ref_index])

        assert ref_structure

        rmsds = np.full(len(model_paths), -1.0)

        if self.process_number > 1:
            with mp.Pool(processes=self.process_number) as pool:
                results = pool.imap_unordered(
                    utils.calculate_rmsd_wrapper,
                    [(ref_structure, m_path, i)
                     for i, m_path in enumerate(model_paths)])

                for result in results:
                    rmsds[result[0]] = result[1]
        else:
            for i, structure in enumerate(model_paths):
                rmsds[i] = utils.calculate_rmsd(ref_structure, structure, i)[1]

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
        plddts = np.empty(len(self.predictions) * len(rank_indeces))

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
        cluster_plddts = np.empty(len(clusters))
        for i, cluster in enumerate(clusters):
            selected_indices = np.where(hbscan.labels_ == cluster)
            cluster_plddts[i] = np.mean(plddts[selected_indices])
            cluster_paths.append(np.unique(model_paths[selected_indices]))

        return cluster_paths, cluster_plddts
