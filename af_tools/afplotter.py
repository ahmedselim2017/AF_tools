import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.figure

import numpy as np

from af_tools.output_types import AFModel, AFPrediction, AF2Model, AF2Prediction, AF3Model, AF3Prediction


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

    def plot_plddt(self, prediction: AFPrediction) -> matplotlib.figure.Figure:

        fig = plt.figure(figsize=self.figsize)
        ax = plt.axes()
        ax.set(ylabel="pLDDT", ylim=(0, 100))

        # AF pLDDT colors
        ax.axhspan(00, 50, facecolor=self.afcolors[0], alpha=0.15)
        ax.axhspan(50, 70, facecolor=self.afcolors[1], alpha=0.15)
        ax.axhspan(70, 90, facecolor=self.afcolors[2], alpha=0.15)
        ax.axhspan(90, 100, facecolor=self.afcolors[3], alpha=0.15)

        if isinstance(prediction, AF3Prediction):
            ax.set_xlabel("Atom")
            for model in reversed(prediction.models):
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

        elif isinstance(prediction, AF2Prediction):
            ax.set_xlabel("Residue")
            for model in reversed(prediction.models):
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

    def plot_pae(self, prediction: AFPrediction) -> matplotlib.figure.Figure:

        fig = plt.figure(figsize=self.figsize)
        cols = len(prediction.models)
        for i, model in enumerate(prediction.models):
            ax = fig.add_subplot(1, cols, i + 1)
            ends: list[int] = []

            if isinstance(model, AF3Model):
                ax.set(ylabel="Token",
                       xlabel="Token",
                       title=f"Rank {model.rank}")
                ends = model.token_chain_ends

            elif isinstance(model, AF2Model):
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
                        predicted_models: list[AFModel],
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
