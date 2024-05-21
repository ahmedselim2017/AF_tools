from dataclasses import dataclass

import json
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.figure
from mpl_toolkits.axes_grid1 import ImageGrid


@dataclass(frozen=True, kw_only=True)
class PredictedModel:
    name: str
    model_path: Path
    json_path: Path
    rank: int
    mean_plddt: float
    ptm: float
    pae: list[float]
    af_version: str


@dataclass(frozen=True, kw_only=True)
class AF2Model(PredictedModel):
    residue_plddt: list[float]
    chain_lengths: list[int]
    is_relaxed: bool


@dataclass(frozen=True, kw_only=True)
class AF3Model(PredictedModel):
    atom_plddts: list[float]
    chain_lengths: list[int]


@dataclass(frozen=True, kw_only=True, slots=True)
class Prediction:
    name: str
    num_ranks: int
    af_version: str
    models_relaxed: list[PredictedModel] | None
    models_unrelaxed: list[PredictedModel]
    is_colabfold: bool


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
            raise Exception(f"Alphafold output directory is not a valid directory: {p}")
        return p

    def get_predictions(self) -> list[Prediction]:
        # Colabfold
        if (self.path / "config.json").is_file():
            return self.get_colabfold_pred()
        # AF2
        elif (self.path / "ranking_debug.json").is_file():
            return self.get_af2_pred()
        # AF3
        elif self.path.glob("*summary_confidences_*.json"):
            return self.get_af3_pred()
        else:
            raise Exception(
                f"Given output directory does not contains Alphafold 2 or Alphafold 3 outputs: {self.path}"
            )

    def get_colabfold_pred(self) -> list[Prediction]:
        with open(self.path / "config.json", "r") as config_file:
            config_data: dict = json.load(config_file)

        af_version: str = config_data["model_type"]
        num_ranks: int = config_data["num_models"]

        preds: list[Prediction] = []
        for pred_done_path in self.path.glob("*.done.txt"):
            pred_name: str = pred_done_path.name.split(".")[0]

            with open(self.path / f"{pred_name}.a3m", "r") as msa_file:
                msa_header_info: list[str] = (
                    msa_file.read().replace("#", "").split("\t")
                )
                msa_header_seq_lengths: list[int] = [
                    int(x) for x in msa_header_info[0].split(",")
                ]
                msa_header_seq_cardinalities: list[int] = [
                    int(x) for x in msa_header_info[1].split(",")
                ]

                chain_lengths: list[int] = []
                for seq_len, seq_cardinality in zip(
                    msa_header_seq_lengths, msa_header_seq_cardinalities
                ):
                    chain_lengths += [seq_len] * seq_cardinality

            model_unrel_paths: list[Path] = sorted(
                self.path.glob(f"{pred_name}_unrelaxed_rank_*.pdb")
            )
            model_rel_paths: list[Path] = sorted(
                self.path.glob(f"{pred_name}_relaxed_rank_*.pdb")
            )
            score_paths: list[Path] = sorted(
                self.path.glob(f"{pred_name}_scores_rank_*.json")
            )

            models_rel: list[PredictedModel] = []
            models_unrel: list[PredictedModel] = []
            for i, (model_unrel_path, score_path) in enumerate(
                zip(model_unrel_paths, score_paths)
            ):
                with open(score_path, "r") as score_file:
                    score_data: dict = json.load(score_file)
                if i < config_data["num_relax"]:
                    model_rel_path: Path = model_rel_paths[i]

                    models_rel.append(
                        AF2Model(
                            name=pred_name,
                            model_path=model_rel_path,
                            json_path=score_path,
                            rank=i + 1,
                            mean_plddt=sum(score_data["plddt"])
                            / len(score_data["plddt"]),
                            ptm=score_data["ptm"],
                            pae=score_data["pae"],
                            af_version=af_version,
                            residue_plddt=score_data["plddt"],
                            chain_lengths=chain_lengths,
                            is_relaxed=True,
                        )
                    )
                models_unrel.append(
                    AF2Model(
                        name=pred_name,
                        model_path=model_unrel_path,
                        json_path=score_path,
                        rank=i + 1,
                        mean_plddt=sum(score_data["plddt"]) / len(score_data["plddt"]),
                        ptm=score_data["ptm"],
                        pae=score_data["pae"],
                        af_version=af_version,
                        residue_plddt=score_data["plddt"],
                        chain_lengths=chain_lengths,
                        is_relaxed=False,
                    )
                )
            preds.append(
                Prediction(
                    name=pred_name,
                    num_ranks=num_ranks,
                    af_version=af_version,
                    models_relaxed=models_rel,
                    models_unrelaxed=models_unrel,
                    is_colabfold=True,
                )
            )
        return preds

    def get_af2_pred(self) -> list[Prediction]:
        return []

    def get_af3_pred(self) -> list[Prediction]:
        return []

    def plot_plddt(
        self, pred: Prediction, is_relaxed_af2: bool = True
    ) -> matplotlib.figure.Figure:
        if not is_relaxed_af2 or pred.af_version == 3:
            assert pred.models_unrelaxed
            models: list[PredictedModel] = pred.models_unrelaxed
        else:
            assert pred.models_relaxed
            models: list[PredictedModel] = pred.models_relaxed

        fig = plt.figure(figsize=self._figsize)
        ax = plt.axes()
        ax.set(ylabel="pLDDT", ylim=(0, 100))
        ax.set_xlabel("Residue") if pred.af_version == 2 else ax.set_xlabel("Atom")

        # AF pLDDT colors
        ax.axhspan(90, 100, facecolor="#106DFF", alpha=0.15)
        ax.axhspan(70, 90, facecolor="#10CFF1", alpha=0.15)
        ax.axhspan(50, 70, facecolor="#F6ED12", alpha=0.15)
        ax.axhspan(00, 50, facecolor="#EF821E", alpha=0.15)

        for model in models:
            if model.af_version == 2:
                ax.plot(
                    range(1, len(model.residue_pLDDT) + 1),  # type:ignore
                    model.residue_pLDDT,
                    color=self._colors[model.rank - 1 % len(self._colors)],
                    label=f"{model.name} Rank {model.rank} Mean pLDDT {model.mean_pLDDT:.3f}",
                )
            # if AF3
            elif model.af_version == 3:
                ax.plot(
                    range(1, len(model.atom_pLDDT) + 1),  # type:ignore
                    model.atom_pLDDT,
                    color=self._colors[model.rank - 1 % len(self._colors)],
                    label=f"{model.name} Rank {model.rank} Mean pLDDT {model.mean_pLDDT:.3f}",
                )
        fig.legend(loc="lower left", bbox_to_anchor=(0.05, 0.07))
        fig.tight_layout()
        return fig

    def plot_all_plddts(
        self, is_relaxed_af2: bool = True
    ) -> list[matplotlib.figure.Figure]:
        figures: list[matplotlib.figure.Figure] = []
        for pred in self.predictions:
            figures.append(self.plot_plddt(pred, is_relaxed_af2))
        return figures

    def plot_pae(
        self, pred: Prediction, is_relaxed_af2: bool = True
    ) -> matplotlib.figure.Figure:

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

        for i, ax in enumerate(grid):  # type: ignore

            ax.set(ylabel="Residue", xlabel="Residue", title=f"Rank {models[i].rank}")

            cax = ax.imshow(models[i].pae, cmap="bwr")
        ax.cax.colorbar(cax)

        return fig
