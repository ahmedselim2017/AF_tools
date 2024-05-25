import multiprocessing as mp
from pathlib import Path
from typing import Sequence

import matplotlib.figure
import numpy as np
from numpy.typing import NDArray

from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Superimposer import Superimposer

from Bio.PDB.Structure import Structure
from Bio.PDB.Atom import Atom

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

    def plot_all_plddts(self) -> list[matplotlib.figure.Figure]:
        figures: list[matplotlib.figure.Figure] = []
        plotter = AFPlotter()
        for pred in self.predictions:
            figures.append(plotter.plot_plddt(pred))
        return figures

    def plot_all_paes(self) -> list[matplotlib.figure.Figure]:
        figures: list[matplotlib.figure.Figure] = []
        plotter = AFPlotter()
        for pred in self.predictions:
            figures.append(plotter.plot_pae(pred))
        return figures

    def plot_plddt_hist(self,
                        use_color: bool = True,
                        draw_mean: bool = True) -> matplotlib.figure.Figure:
        plotter = AFPlotter()

        predicted_models: list[AFModel] = []
        for pred in self.predictions:
            predicted_models += pred.models

        fig = plotter.plot_plddt_hist(predicted_models=predicted_models,
                                      use_color=use_color,
                                      draw_mean=draw_mean)

        return fig

    def _calculate_rmsds(self, model_paths: list[Path], ref_pred_index: int,
                         rank_index: int) -> NDArray:

        ref_structure: Structure = utils.load_structure(
            model_paths[ref_pred_index])

        rmsds = np.full(len(self.predictions), -1.0)
        rmsds[ref_pred_index] = 0.0

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

        return rmsds

    def plot_rmsds(self, ref_pred_index: int,
                   rank_index: int) -> matplotlib.figure.Figure:
        model_paths: list[Path] = []
        model_plddts: list[float] = []

        for pred in self.predictions:
            model = pred.models[rank_index]
            model_plddts.append(model.mean_plddt)

            if hasattr(model, "relaxed_pdb_path"):
                model_paths.append(model.relaxed_pdb_path)
            else:
                model_paths.append(model.model_path)

        rmsds = self._calculate_rmsds(model_paths=model_paths,
                                      ref_pred_index=ref_pred_index,
                                      rank_index=rank_index)

        plotter = AFPlotter()
        return plotter.plot_rmsd_plddt(model_plddts, rmsds)
