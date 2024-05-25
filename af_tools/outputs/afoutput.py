import multiprocessing
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


class AFOutput:

    def __init__(self, path: str | Path, process_number: int = 1):
        self.path = self.check_path(path)
        self.process_number = process_number
        self.predictions = self.get_predictions()

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

    def _calculate_rmsd(self,
                        ref_model_path: Path,
                        target_model_path: Path,
                        index: int = 0) -> tuple[int, float]:

        if ref_model_path == target_model_path:
            return (index, 0)

        models: list[Structure] = []

        if ref_model_path.suffix == target_model_path.suffix and ref_model_path.suffix == ".cif":
            parser = MMCIFParser()
            models.append(
                parser.get_structure(ref_model_path.name, ref_model_path))
            models.append(
                parser.get_structure(target_model_path.name,
                                     target_model_path))
        elif ref_model_path.suffix == target_model_path.suffix and ref_model_path.suffix == ".pdb":
            parser = PDBParser()
            models.append(
                parser.get_structure(ref_model_path.name, ref_model_path))
            models.append(
                parser.get_structure(target_model_path.name,
                                     target_model_path))
        elif ref_model_path.suffix != target_model_path.suffix:
            for model_path in (ref_model_path, target_model_path):
                if model_path.suffix == ".cif":
                    parser = MMCIFParser()
                    models.append(
                        parser.get_structure(model_path.name, model_path))
                elif model_path.suffix == ".pdb":
                    parser = PDBParser()
                    models.append(
                        parser.get_structure(model_path.name, model_path))
                else:
                    raise Exception(f"Unknwon model file type:{model_path}")
        else:
            raise Exception(
                f"Unknwon model file types:{ref_model_path} and {target_model_path}"
            )

        m1_cas: list[Atom] = []
        m2_cas: list[Atom] = []
        for m1_res, m2_res in zip(models[0].get_residues(),
                                  models[1].get_residues()):
            assert m1_res.resname == m2_res.resname
            assert m1_res.id == m2_res.id

            m1_cas.append(m1_res["CA"])
            m2_cas.append(m2_res["CA"])

        sup = Superimposer()
        sup.set_atoms(m1_cas, m2_cas)

        assert isinstance(sup.rms, float)

        return (index, sup.rms)

    def calculate_rmsds(self, ref_pred_index: int, rank_index: int) -> NDArray:

        model_paths: list[Path] = []
        ref_model_path: Path | None = None
        for i, pred in enumerate(self.predictions):
            model = pred.models[rank_index]
            if hasattr(model, "relaxed_pdb_path"):
                model_paths.append(model.relaxed_pdb_path)
                if i == ref_pred_index:
                    ref_model_path = model.relaxed_pdb_path
            else:
                model_paths.append(model.model_path)
                if i == ref_pred_index:
                    ref_model_path = model.model_path
        assert ref_model_path is not None

        rmsds = np.full(len(self.predictions), -1.0)
        rmsds[ref_pred_index] = 0.0

        if self.process_number > 1:
            with multiprocessing.Pool(processes=self.process_number) as pool:
                results = pool.starmap(
                    self._calculate_rmsd,
                    [(ref_model_path, m_path, i)
                     for i, m_path in enumerate(model_paths)])
                for result in results:
                    rmsds[result[0]] = result[1]
        else:
            for i, target_model_path in enumerate(model_paths):
                rmsds[i] = self._calculate_rmsd(ref_model_path,
                                                target_model_path, i)

        return rmsds
