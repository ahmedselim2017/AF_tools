from dataclasses import dataclass
import multiprocessing

import orjson  # type:ignore
from pathlib import Path  # type:ignore
import numpy as np  # type:ignore
from numpy.typing import NDArray  # type:ignore

import matplotlib.pyplot as plt  # type:ignore
import matplotlib.colors as mcolors  # type:ignore
import matplotlib.figure  # type:ignore


@dataclass(frozen=True, kw_only=True, slots=True)
class PredictedModel:
    name: str
    model_path: Path
    json_path: Path
    rank: int
    mean_plddt: float
    ptm: float
    pae: NDArray
    af_version: str


@dataclass(frozen=True, kw_only=True, slots=True)
class AF2Model(PredictedModel):
    residue_plddts: NDArray
    chain_ends: list[int]
    relaxed_pdb_path: Path | None


@dataclass(frozen=True, kw_only=True, slots=True)
class AF3Model(PredictedModel):
    atom_plddts: NDArray
    atom_chain_ends: list[int]
    token_chain_ends: list[int]


@dataclass(frozen=True, kw_only=True, slots=True)
class Prediction:
    name: str
    num_ranks: int
    af_version: str
    models: list[PredictedModel]
    is_colabfold: bool


class AFOutput:

    def __init__(self,
                 path: str | Path,
                 search_recursively: bool = False,
                 process_number: int = 1):
        self.path = self.check_path(path)
        self.search_recursively = search_recursively
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

    def get_predictions(self) -> list[Prediction]:
        # Colabfold
        if (self.path / "config.json").is_file():
            return self.get_colabfold_pred()
        # AF2
        elif (self.path / "ranking_debug.json").is_file():
            return self.get_af2_pred()
        # AF3
        elif any(True for _ in self.path.glob("*summary_confidences_*.json")):
            return self.get_af3_pred()
        # Only works for colabfold for now
        elif len(list(self.path.rglob("config.json"))) > 1:
            return self.get_pred_recursively()
        else:
            raise Exception(
                f"Given output directory does not contains Alphafold 2 or Alphafold 3 outputs: {self.path}"
            )

    def get_pred_worker(self, path: Path):
        afoutput = AFOutput(path=path)
        return afoutput.predictions

    def get_pred_recursively(self) -> list[Prediction]:
        outputs = [x.parent for x in list(self.path.rglob("config.json"))]
        with multiprocessing.Pool(processes=self.process_number) as pool:
            results = pool.map(self.get_pred_worker, outputs)
        predictions: list[Prediction] = [j for i in results
                                         for j in i]  # flatten the results

        return predictions

    def get_colabfold_pred(self) -> list[Prediction]:
        with open(self.path / "config.json", "rb") as config_file:
            config_data = orjson.loads(config_file.read())

        af_version = config_data["model_type"]
        num_ranks = config_data["num_models"]

        preds: list[Prediction] = []
        for pred_done_path in self.path.glob("*.done.txt"):
            pred_name = pred_done_path.name.split(".")[0]

            with open(self.path / f"{pred_name}.a3m", "r") as msa_file:
                msa_header_info = msa_file.readline().replace("#",
                                                              "").split("\t")
            msa_header_seq_lengths = [
                int(x) for x in msa_header_info[0].split(",")
            ]
            msa_header_seq_cardinalities = [
                int(x) for x in msa_header_info[1].split(",")
            ]

            chain_lengths: list[int] = []
            for seq_len, seq_cardinality in zip(msa_header_seq_lengths,
                                                msa_header_seq_cardinalities):
                chain_lengths += [seq_len] * seq_cardinality

            chain_ends: list[int] = []
            for chain_len in chain_lengths:
                if chain_ends == []:
                    chain_ends.append(chain_len)
                else:
                    chain_ends.append(chain_len + chain_ends[-1])

            model_unrel_paths = sorted(
                self.path.glob(f"{pred_name}_unrelaxed_rank_*.pdb"))
            model_rel_paths = sorted(
                self.path.glob(f"{pred_name}_relaxed_rank_*.pdb"))
            score_paths = sorted(
                self.path.glob(f"{pred_name}_scores_rank_*.json"))

            models: list[PredictedModel] = []
            for i, (model_unrel_path, score_path) in enumerate(
                    zip(model_unrel_paths, score_paths)):
                model_rel_path = None
                if i < config_data["num_relax"]:
                    model_rel_path = model_rel_paths[i]

                with open(score_path, "rb") as score_file:
                    score_data = orjson.loads(score_file.read())
                pae = np.asarray(score_data["pae"])
                plddt = np.asarray(score_data["plddt"])

                models.append(
                    AF2Model(
                        name=pred_name,
                        model_path=model_unrel_path,
                        relaxed_pdb_path=model_rel_path,
                        json_path=score_path,
                        rank=i + 1,
                        mean_plddt=np.average(plddt, axis=0),
                        ptm=score_data["ptm"],
                        pae=pae,
                        af_version=af_version,
                        residue_plddts=plddt,
                        chain_ends=chain_ends,
                    ))
            preds.append(
                Prediction(
                    name=pred_name,
                    num_ranks=num_ranks,
                    af_version=af_version,
                    models=models,
                    is_colabfold=True,
                ))
        return preds

    def get_af2_pred(self) -> list[Prediction]:
        return []

    def get_af3_pred(self) -> list[Prediction]:
        af_version = "alphafold3"
        full_data_paths = sorted(self.path.glob("*_full_data*.json"))
        summary_data_paths = sorted(
            self.path.glob("*summary_confidences_*.json"))
        model_paths = sorted(self.path.glob("*.cif"))

        pred_name = full_data_paths[0].name.split("_full_data_")[0]
        models: list[PredictedModel] = []
        for i, (full_data_path, summary_data_path, model_path) in enumerate(
                zip(full_data_paths, summary_data_paths, model_paths)):
            with open(full_data_path,
                      "rb") as full_data_file, open(summary_data_path,
                                                    "rb") as summary_data_file:
                full_data = orjson.loads(full_data_file.read())
                summary_data = orjson.loads(summary_data_file.read())

            atom_chain_lengths: list[int] = []
            atom_id_old = ""
            chain_length = 0
            for atom_id in full_data["atom_chain_ids"]:
                if atom_id != atom_id_old:
                    if atom_id_old != "":
                        atom_chain_lengths.append(chain_length)
                    chain_length = 1
                else:
                    chain_length += 1
                atom_id_old = atom_id
            atom_chain_lengths.append(chain_length)

            atom_chain_ends: list[int] = []
            for chain_len in atom_chain_lengths:
                if atom_chain_ends == []:
                    atom_chain_ends.append(chain_len)
                else:
                    atom_chain_ends.append(chain_len + atom_chain_ends[-1])

            token_chain_lengths: list[int] = []
            token_id_old = ""
            chain_length = 0
            for token_id in full_data["token_chain_ids"]:
                if token_id != token_id_old:
                    if token_id_old != "":
                        token_chain_lengths.append(chain_length)
                    chain_length = 1
                else:
                    chain_length += 1
                token_id_old = token_id
            token_chain_lengths.append(chain_length)

            token_chain_ends: list[int] = []
            for chain_len in token_chain_lengths:
                if token_chain_ends == []:
                    token_chain_ends.append(chain_len)
                else:
                    token_chain_ends.append(chain_len + token_chain_ends[-1])

            models.append(
                AF3Model(
                    name=pred_name,
                    model_path=model_path,
                    json_path=full_data_path,
                    rank=i + 1,
                    mean_plddt=sum(full_data["atom_plddts"]) /
                    len(full_data["atom_plddts"]),
                    ptm=summary_data["ptm"],
                    pae=full_data["pae"],
                    af_version=af_version,
                    atom_plddts=full_data["atom_plddts"],
                    atom_chain_ends=atom_chain_ends,
                    token_chain_ends=token_chain_ends,
                ))
        return [
            Prediction(
                name=pred_name,
                num_ranks=len(models),
                af_version=af_version,
                models=models,
                is_colabfold=False,
            )
        ]

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

        predicted_models: list[PredictedModel] = []
        for pred in self.predictions:
            predicted_models += pred.models

        fig = plotter.plot_plddt_hist(predicted_models=predicted_models,
                                      use_color=use_color,
                                      draw_mean=draw_mean)

        return fig


class AFPlotter:

    def __init__(
        self,
        figsize: tuple[float, float] | None = None,
        colors: list[str] | None = None,
    ):
        if figsize:
            self.figsize = figsize
        else:
            px = 1 / plt.rcParams["figure.dpi"]
            self.figsize = (1618 * px, 1000 * px)

        if colors:
            self.colors = colors
        else:
            self.colors = list(mcolors.TABLEAU_COLORS.values())

        self.afcolors = ["#FF7D45", "#FFDB13", "#65CBF3", "#0053D6"]

    def plot_plddt(self, pred) -> matplotlib.figure.Figure:

        fig = plt.figure(figsize=self.figsize)
        ax = plt.axes()
        ax.set(ylabel="pLDDT", ylim=(0, 100))

        # AF pLDDT colors
        ax.axhspan(00, 50, facecolor=self.afcolors[0], alpha=0.15)
        ax.axhspan(50, 70, facecolor=self.afcolors[1], alpha=0.15)
        ax.axhspan(70, 90, facecolor=self.afcolors[2], alpha=0.15)
        ax.axhspan(90, 100, facecolor=self.afcolors[3], alpha=0.15)

        if pred.af_version == "alphafold3":
            ax.set_xlabel("Atom")
            for model in reversed(pred.models):
                assert isinstance(model, AF3Model)
                ax.plot(
                    range(1,
                          len(model.atom_plddts) + 1),
                    model.atom_plddts,
                    color=self.colors[model.rank - 1 % len(self.colors)],
                    label=
                    f"{model.name} Rank {model.rank} Mean pLDDT {model.mean_plddt:.3f}",
                )

                if len(model.atom_chain_ends) > 1:
                    ax.vlines(
                        model.atom_chain_ends[:-1],
                        ymin=0,
                        ymax=100,
                        color="black",
                    )

        elif "alphafold2" in pred.af_version:
            ax.set_xlabel("Residue")
            for model in reversed(pred.models):
                assert isinstance(model, AF2Model)
                ax.plot(
                    range(1,
                          len(model.residue_plddts) + 1),
                    model.residue_plddts,
                    color=self.colors[model.rank - 1 % len(self.colors)],
                    label=
                    f"{model.name} Rank {model.rank} Mean pLDDT {model.mean_plddt:.3f}",
                )

                if len(model.chain_ends) > 1:
                    ax.vlines(model.chain_ends[:-1],
                              ymin=0,
                              ymax=100,
                              color="black")

        fig.legend(loc="lower left", bbox_to_anchor=(0.05, 0.07))
        fig.tight_layout()
        return fig

    def plot_pae(self, pred: Prediction) -> matplotlib.figure.Figure:

        fig = plt.figure(figsize=self.figsize)
        cols = len(pred.models)
        for i, model in enumerate(pred.models):
            ax = fig.add_subplot(1, cols, i + 1)
            ends: list[int] = []

            if pred.af_version == "alphafold3":
                ax.set(ylabel="Token",
                       xlabel="Token",
                       title=f"Rank {model.rank}")
                assert isinstance(model, AF3Model)
                ends = model.token_chain_ends

            elif "alphafold2" in pred.af_version:
                ax.set(ylabel="Residue",
                       xlabel="Residue",
                       title=f"Rank {model.rank}")
                assert isinstance(model, AF2Model)
                ends = model.chain_ends

            cax = ax.matshow(
                model.pae,
                cmap="bwr",
                extent=(0.5, len(model.pae) - 0.5, len(model.pae) - 0.5, 0.5),
            )
            fig.colorbar(cax, fraction=0.046, pad=0.04)

            if len(ends) > 1:
                ax.vlines(
                    ends[:-1],
                    ymin=0.5,
                    ymax=len(model.pae) - 0.5,
                    color="black",
                    linewidth=2,
                )
                ax.hlines(
                    ends[:-1],
                    xmin=0.5,
                    xmax=len(model.pae) - 0.5,
                    color="black",
                    linewidth=2,
                )

            ax.tick_params(axis="x", labelrotation=90)
        fig.tight_layout()
        return fig

    def plot_plddt_hist(self,
                        predicted_models: list[PredictedModel],
                        use_color: bool = True,
                        draw_mean: bool = True) -> matplotlib.figure.Figure:

        mean_plddts = np.zeros(len(predicted_models))
        for i, pred in enumerate(predicted_models):
            mean_plddts[i] = pred.mean_plddt

        fig = plt.figure(figsize=self.figsize)
        ax = plt.axes()
        ax.set(xlabel="pLDDT", ylabel="Count", xlim=(0, 100))

        if use_color:
            ax.axvspan(00, 50, facecolor=self.afcolors[0], alpha=0.15)
            ax.axvspan(50, 70, facecolor=self.afcolors[1], alpha=0.15)
            ax.axvspan(70, 90, facecolor=self.afcolors[2], alpha=0.15)
            ax.axvspan(90, 100, facecolor=self.afcolors[3], alpha=0.15)

        if draw_mean:
            mean = mean_plddts.mean()
            ax.axvline(mean,
                       color="k",
                       linestyle="dashed",
                       label=f"Mean: {mean:.3f}")

        ax.hist(mean_plddts,
                bins=30,
                color="tab:blue",
                alpha=0.75,
                edgecolor="k")

        ax.legend()
        fig.tight_layout()
        return fig
