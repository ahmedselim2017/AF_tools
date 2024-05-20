from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.figure
from mpl_toolkits.axes_grid1 import ImageGrid


@dataclass(frozen=True, kw_only=True, slots=True)
class PredictedModel:
    name: str
    model_path: Path
    json_path: Path
    rank: int
    mean_pLDDT: float
    pTM: float
    pae: np.ndarray
    atom_pLDDT: np.ndarray | None
    residue_pLDDT: np.ndarray | None
    af_version: int
    is_relaxed: bool | None


@dataclass(frozen=True, kw_only=True, slots=True)
class Prediction:
    name: str
    num_ranks: int
    path: Path
    af_version: int
    models_relaxed: list[PredictedModel] | None
    models_unrelaxed: list[PredictedModel]


class AFOutput:

    def __init__(self, path: str):
        self.path: Path = self.check_path(path)
        self.predictions = self.get_predictions()

        self._colors = list(mcolors.TABLEAU_COLORS.values())
        px = 1 / plt.rcParams["figure.dpi"]
        self._figsize = (1618 * px, 1000 * px)

    def check_path(self, path) -> Path:
        p: Path = Path(path)
        if not p.is_dir():
            raise Exception(
                f"Alphafold output directory is not a valid directory: {p}")
        return p

    def get_predictions(self) -> list[Prediction]:
        predictions: list[Prediction] = []
        for af2_pred_done in self.path.glob("*.done.txt"):
            name: str = af2_pred_done.name.split(".")[0]

            unrel_model_paths: list[Path] = sorted(
                self.path.glob(f"{name}_unrelaxed_rank_*pdb"))
            model_fulljson_paths: list[Path] = sorted(
                self.path.glob(f"{name}_scores_rank_*json"))

            unrelaxed_models: list[PredictedModel] = []
            relaxed_models: list[PredictedModel] = []
            for i, (unrel_model_path, model_fulljson_path) in enumerate(
                    zip(unrel_model_paths, model_fulljson_paths)):

                with open(model_fulljson_path, "r") as model_fulljson_file:
                    model_json: dict = json.load(model_fulljson_file)

                np_plddt: NDArray = np.asarray(model_json["plddt"])
                np_pae: NDArray = np.asarray(model_json["pae"])
                unrelaxed_models.append(
                    PredictedModel(name=name,
                                   model_path=unrel_model_path,
                                   json_path=model_fulljson_path,
                                   rank=i + 1,
                                   mean_pLDDT=np.average(np_plddt, axis=0),
                                   pTM=model_json["ptm"],
                                   pae=np_pae,
                                   atom_pLDDT=None,
                                   residue_pLDDT=np_plddt,
                                   af_version=2,
                                   is_relaxed=False))

            rel_model_paths: list[Path] = sorted(
                self.path.glob(f"{name}_relaxed_rank_*pdb"))
            for i, rel_model_path in enumerate(rel_model_paths):

                model_fulljson_path: Path = next(
                    self.path.glob(
                        f"{name}_scores_rank_{i+1:03d}_alphafold2*json"))
                with open(model_fulljson_path, "r") as model_fulljson_file:
                    model_json: dict = json.load(model_fulljson_file)

                np_plddt = np.asarray(model_json["plddt"])
                np_pae = np.asarray(model_json["pae"])
                relaxed_models.append(
                    PredictedModel(name=name,
                                   model_path=rel_model_path,
                                   json_path=model_fulljson_path,
                                   rank=i + 1,
                                   mean_pLDDT=np.average(np_plddt, axis=0),
                                   pTM=model_json["ptm"],
                                   pae=np_pae,
                                   atom_pLDDT=None,
                                   residue_pLDDT=np_plddt,
                                   af_version=2,
                                   is_relaxed=False))
            predictions.append(
                Prediction(
                    name=name,
                    num_ranks=len(unrelaxed_models),
                    path=self.path,
                    af_version=2,
                    models_relaxed=relaxed_models,
                    models_unrelaxed=unrelaxed_models,
                ))

        if predictions:
            return predictions

        if (self.path / "terms_of_use.md").is_file():
            model_paths = sorted(self.path.glob("*model*.cif"))
            model_fulljson_paths = sorted(self.path.glob("*full_data_*.json"))
            model_summaryjson_paths = sorted(
                self.path.glob("*summary_confidences*.json"))

            predicted_models: list[PredictedModel] = []
            for i, (model_path, model_fulljson_path,
                    model_summaryjson_path) in enumerate(
                        zip(model_paths, model_fulljson_paths,
                            model_summaryjson_paths)):
                with open(model_fulljson_path, "r") as model_fulljson_file:
                    model_fulldata: dict = json.load(model_fulljson_file)
                with open(model_summaryjson_path,
                          "r") as model_summaryjson_file:
                    model_summarydata: dict = json.load(model_summaryjson_file)

                name: str = model_fulljson_path.name.split("_full_data_")[0]
                np_plddt: NDArray = np.asarray(model_fulldata["atom_plddts"])
                np_pae: NDArray = np.asarray(model_fulldata["pae"])
                predicted_models.append(
                    PredictedModel(name=name,
                                   model_path=model_path,
                                   json_path=model_fulljson_path,
                                   rank=i + 1,
                                   mean_pLDDT=np.average(np_plddt, axis=0),
                                   pTM=model_summarydata["ptm"],
                                   pae=np_pae,
                                   atom_pLDDT=np_plddt,
                                   residue_pLDDT=None,
                                   af_version=3,
                                   is_relaxed=None))

            predictions.append(
                Prediction(
                    name=name,  # type: ignore
                    num_ranks=len(predicted_models),
                    path=self.path,
                    af_version=3,
                    models_relaxed=None,
                    models_unrelaxed=predicted_models))
            return predictions
        else:
            raise Exception(
                f"Given output directory does not contains Alphafold 2 or Alphafold 3 outputs: {self.path}"
            )

    def plot_plddt(self,
                   pred: Prediction,
                   is_relaxed_af2: bool = True) -> matplotlib.figure.Figure:
        if not is_relaxed_af2 or pred.af_version == 3:
            assert pred.models_unrelaxed
            models: list[PredictedModel] = pred.models_unrelaxed
        else:
            assert pred.models_relaxed
            models: list[PredictedModel] = pred.models_relaxed

        fig = plt.figure(figsize=self._figsize)
        ax = plt.axes()
        ax.set(ylabel="pLDDT", ylim=(0, 100))
        ax.set_xlabel("Residue") if pred.af_version == 2 else ax.set_xlabel(
            "Atom")

        # AF pLDDT colors
        ax.axhspan(90, 100, facecolor="#106DFF", alpha=0.15)
        ax.axhspan(70, 90, facecolor="#10CFF1", alpha=0.15)
        ax.axhspan(50, 70, facecolor="#F6ED12", alpha=0.15)
        ax.axhspan(00, 50, facecolor="#EF821E", alpha=0.15)

        for model in models:
            if model.af_version == 2:
                ax.plot(
                    range(1,
                          len(model.residue_pLDDT) + 1),
                    model.residue_pLDDT,
                    color=self._colors[model.rank - 1 % len(self._colors)],
                    label=
                    f"{model.name} Rank {model.rank} Mean pLDDT {model.mean_pLDDT:.3f}"
                )
            # if AF3
            elif model.af_version == 3:
                ax.plot(
                    range(1,
                          len(model.atom_pLDDT) + 1),
                    model.atom_pLDDT,
                    color=self._colors[model.rank - 1 % len(self._colors)],
                    label=
                    f"{model.name} Rank {model.rank} Mean pLDDT {model.mean_pLDDT:.3f}"
                )
        fig.legend(loc="lower left", bbox_to_anchor=(0.05, 0.07))
        fig.tight_layout()
        return fig

    def plot_all_plddts(
            self,
            is_relaxed_af2: bool = True) -> list[matplotlib.figure.Figure]:
        figures: list[matplotlib.figure.Figure] = []
        for pred in self.predictions:
            figures.append(self.plot_plddt(pred, is_relaxed_af2))
        return figures

    def plot_pae(self,
                 pred: Prediction,
                 is_relaxed_af2: bool = True) -> matplotlib.figure.Figure:

        if not is_relaxed_af2 or pred.af_version == 3:
            assert pred.models_unrelaxed
            models: list[PredictedModel] = pred.models_unrelaxed
        else:
            assert pred.models_relaxed
            models: list[PredictedModel] = pred.models_relaxed

        fig = plt.figure(figsize=self._figsize)
        grid = ImageGrid(
            fig,
            111,
            nrows_ncols=(1, len(models)),
            axes_pad=0.1,
            cbar_location="right",
            cbar_mode="single",
            cbar_size="7%",
            cbar_pad=0.15,
        )

        for i, ax in enumerate(grid):

            ax.set(ylabel="Residue",
                   xlabel="Residue",
                   title=f"Rank {models[i].rank}")

            cax = ax.imshow(models[i].pae, cmap="bwr")
        ax.cax.colorbar(cax)

        return fig
