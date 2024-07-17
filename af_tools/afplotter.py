import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
import matplotlib.figure

from pandas import DataFrame
import seaborn as sns

import numpy as np
from numpy.typing import NDArray

from natsort import natsort_keygen

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
            self.colors = sns.color_palette()

        self.afcolors = ["#FF7D45", "#FFDB13", "#65CBF3", "#0053D6"]

    def plot_plddt(self, df: DataFrame) -> matplotlib.figure.Figure:

        df_sorted = df.sort_values(by="best_model_path", key=natsort_keygen())
        df_sorted = df_sorted.reset_index()

        fig = plt.figure(figsize=self.figsize)
        ax = plt.axes()
        ax.set(ylabel="pLDDT", ylim=(0, 100))

        # AF pLDDT colors
        ax.axhspan(00, 50, facecolor=self.afcolors[0], alpha=0.15)
        ax.axhspan(50, 70, facecolor=self.afcolors[1], alpha=0.15)
        ax.axhspan(70, 90, facecolor=self.afcolors[2], alpha=0.15)
        ax.axhspan(90, 100, facecolor=self.afcolors[3], alpha=0.15)

        sns.lineplot(ax=ax, data=df_sorted, y="plddt", hue="index")

        fig.legend(loc="lower left", bbox_to_anchor=(0.05, 0.07), reverse=True)
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

    def plot_plddt_hist(
            self,
            predicted_models: list[AFModel],
            use_color: bool = True,
            draw_mean: bool = True,
            xlim: tuple[float, float] | None = None
    ) -> matplotlib.figure.Figure:

        mean_plddts = np.zeros(len(predicted_models))
        for i, pred in enumerate(predicted_models):
            mean_plddts[i] = pred.mean_plddt

        fig = plt.figure(figsize=self.figsize)
        ax = plt.axes()

        ax.hist(mean_plddts,
                bins=np.histogram_bin_edges(mean_plddts),
                color="tab:blue",
                alpha=0.75,
                edgecolor="k")

        ax.set(xlabel="pLDDT", ylabel="Count", xlim=xlim)

        if draw_mean:
            mean = mean_plddts.mean()
            ax.axvline(mean,
                       color="k",
                       linestyle="dashed",
                       label=f"Mean: {mean:.3f}")
        if use_color:
            ax.axvspan(00, 50, facecolor=self.afcolors[0], alpha=0.15)
            ax.axvspan(50, 70, facecolor=self.afcolors[1], alpha=0.15)
            ax.axvspan(70, 90, facecolor=self.afcolors[2], alpha=0.15)
            ax.axvspan(90, 100, facecolor=self.afcolors[3], alpha=0.15)

        ax.legend()
        fig.tight_layout()
        return fig

    def plot_upper_trig(self,
                        matrix: NDArray,
                        labels: list[str] | None = None,
                        log_scale: bool = False,
                        vmin: float | None = None,
                        vmax: float | None = None) -> matplotlib.figure.Figure:
        fig = plt.figure(figsize=(min(self.figsize), min(self.figsize)))
        ax = plt.axes()

        ax.yaxis.tick_right()
        ax.spines[['left', 'bottom']].set_visible(False)
        ax.grid(False)

        ax.tick_params(axis="x", bottom=False)
        ax.tick_params(axis="y", left=False)

        mask = np.tril(np.ones((matrix.shape[0], matrix.shape[0])))
        np.fill_diagonal(mask, 0)
        matrix = np.ma.array(matrix, mask=mask)

        cmap = matplotlib.colormaps["plasma"]
        cmap.set_bad('w')
        if log_scale:
            cax = ax.matshow(matrix,
                             cmap=cmap,
                             norm=LogNorm(),
                             vmin=vmin,
                             vmax=vmax)
        else:
            cax = ax.matshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)

        fig.colorbar(cax, fraction=0.046, pad=0.04, location="left")

        if labels:
            xaxis = np.arange(len(labels))
            ax.set_xticks(xaxis)
            ax.set_yticks(xaxis)
            ax.set_xticklabels(labels, rotation=90)
            ax.set_yticklabels(labels)

        return fig

    def plot_data_plddt(
            self,
            rmsds: NDArray,
            mean_plddts: NDArray,
            datalabel: str,
            labels: NDArray | None = None) -> matplotlib.figure.Figure:

        fig = plt.figure(figsize=self.figsize)
        ax = plt.axes()

        ax.set(xlabel=datalabel, ylabel="pLDDT", ylim=(0, 100))

        ax.axhspan(00, 50, facecolor=self.afcolors[0], alpha=0.15)
        ax.axhspan(50, 70, facecolor=self.afcolors[1], alpha=0.15)
        ax.axhspan(70, 90, facecolor=self.afcolors[2], alpha=0.15)
        ax.axhspan(90, 100, facecolor=self.afcolors[3], alpha=0.15)

        if labels is None:
            ax.scatter(rmsds, mean_plddts, alpha=0.3)
        else:
            for i, label in enumerate(np.unique(labels)):
                color = self.colors[i % len(
                    self.colors)] if label != -1 else "black"

                selected_indices = np.where(labels == label)

                ax.scatter(rmsds[selected_indices],
                           mean_plddts[selected_indices],
                           alpha=0.3,
                           label=label,
                           color=color)

            ax.legend()

        fig.tight_layout()
        return fig

    def plot_data_conf(
            self,
            rmsds: NDArray,
            confs: NDArray,
            datalabel: str,
            labels: NDArray | None = None) -> matplotlib.figure.Figure:

        fig = plt.figure(figsize=self.figsize)
        ax = plt.axes()

        ax.set(xlabel=datalabel, ylabel="Confidence", ylim=(0, 1))

        if labels is None:
            ax.scatter(rmsds, confs, alpha=0.3)
        else:
            for i, label in enumerate(np.unique(labels)):
                color = self.colors[i % len(
                    self.colors)] if label != -1 else "black"

                selected_indices = np.where(labels == label)

                ax.scatter(rmsds[selected_indices],
                           confs[selected_indices],
                           alpha=0.3,
                           label=label,
                           color=color)

            ax.legend()

        fig.tight_layout()
        return fig
