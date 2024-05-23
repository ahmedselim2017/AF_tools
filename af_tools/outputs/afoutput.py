from pathlib import Path
from typing import Sequence

import matplotlib.figure

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
