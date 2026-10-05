import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from src.database_service import AXES
from src.regression_models import AxisRegression


class ResidualAnalyzer:

    def __init__(self, models, frame, max_gap=10.0):
        self.models = models
        self.frame = frame.reset_index(drop=True)
        self.residuals = models.residuals(self.frame)
        self.mask = models.running_mask(self.frame)
        self.active = self.residuals[self.mask]
        self.seconds = models.seconds(self.frame["Timestamp"])
        self.max_gap = max_gap

    def run_lengths(self, column, k):
        above = self.residuals[column].to_numpy() >= k * self.models.sigma[column]
        previous = np.r_[False, above[:-1]]
        gap = np.r_[False, np.diff(self.seconds) > self.max_gap]
        run_id = np.cumsum(above & (~previous | gap))[above]
        if run_id.size == 0:
            return np.array([])
        times = pd.Series(self.seconds[above])
        grouped = times.groupby(run_id)
        return (grouped.max() - grouped.min()).to_numpy()

    def outlier_table(self, ks=(1, 2, 3, 4, 5), share=False):
        rows = {}
        for column in AXES:
            sigma = self.models.sigma[column]
            counts = {f">= {k} sigma": (self.active[column] >= k * sigma) for k in ks}
            rows[self.models.label(column)] = {
                key: (float(v.mean() * 100) if share else int(v.sum())) for key, v in counts.items()
            }
        return pd.DataFrame(rows).T

    def state_summary(self):
        running = int(self.mask.sum())
        return pd.Series({
            "readings": len(self.frame),
            "running": running,
            "idle": len(self.frame) - running,
            "running_%": round(running / len(self.frame) * 100, 1),
            "idle_%": round((1 - running / len(self.frame)) * 100, 1),
        })

    @staticmethod
    def plot_state_share(frame, save=None):
        mask = AxisRegression.running_mask(frame)
        hourly = mask.groupby(frame["Timestamp"].dt.floor("h")).mean() * 100
        fig, (left, right) = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={"width_ratios": [1, 2.4]})
        share = mask.mean() * 100
        left.bar(["Running", "Idle (all axes 0)"], [share, 100 - share], color=["tab:green", "lightgray"])
        left.set_ylabel("% of readings")
        left.set_title("Machine state")
        for i, v in enumerate([share, 100 - share]):
            left.text(i, v + 1, f"{v:.1f}%", ha="center")
        right.bar(hourly.index, hourly.values, width=0.03, color="tab:green")
        right.set_ylabel("% running")
        right.set_title("Share of each hour the machine was running")
        right.xaxis.set_major_formatter(mdates.DateFormatter("%d %H:%M"))
        fig.tight_layout()
        return ResidualAnalyzer.finish(fig, save)

    def longest_runs(self, ks=(1, 2, 3, 4)):
        rows = {}
        for column in AXES:
            rows[self.models.label(column)] = {
                f"{k} sigma": float(self.run_lengths(column, k).max(initial=0)) for k in ks
            }
        return pd.DataFrame(rows).T

    def event_grid(self, ks=(1, 1.5, 2, 3, 4), durations=(10, 20, 30, 45, 60)):
        grid = pd.DataFrame(index=[f"{k} sigma" for k in ks], columns=[f"T={d}s" for d in durations])
        for k in ks:
            lengths = np.concatenate([self.run_lengths(c, k) for c in AXES])
            for d in durations:
                grid.loc[f"{k} sigma", f"T={d}s"] = int((lengths >= d).sum())
        return grid.astype(int)

    def plot_regressions(self, points=4000, save=None):
        running = self.frame[self.mask]
        step = max(1, len(running) // points)
        sample = running.iloc[::step]
        fig, axes = plt.subplots(4, 2, figsize=(15, 12), sharex=True)
        for ax, column in zip(axes.ravel(), AXES):
            row = self.models.table.loc[self.models.label(column)]
            ax.scatter(sample["Timestamp"], sample[column], s=3, alpha=0.35, color="tab:blue")
            ax.plot(sample["Timestamp"], self.models.predict(sample["Timestamp"], column),
                    color="red", linewidth=2)
            ax.set_title(f"{self.models.label(column)}  slope {row['slope_per_hour']:+.4f}/h  "
                         f"intercept {row['intercept']:.3f}  R2 {row['r2']:.4f}", fontsize=9)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        fig.suptitle("Univariate linear regression: time to current, running readings only")
        fig.tight_layout()
        return self.finish(fig, save)

    def plot_residual_histograms(self, save=None):
        fig, axes = plt.subplots(2, 4, figsize=(16, 7))
        for ax, column in zip(axes.ravel(), AXES):
            residual = self.active[column]
            sigma = self.models.sigma[column]
            ax.hist(residual, bins=80, color="tab:blue", alpha=0.8)
            ax.set_yscale("log")
            for k, color in ((2, "orange"), (4, "red")):
                ax.axvline(k * sigma, color=color, linestyle="--", linewidth=1)
            ax.set_title(self.models.label(column), fontsize=10)
        fig.suptitle("Residual distributions, running readings only (log count), dashed lines at 2 and 4 sigma")
        fig.tight_layout()
        return self.finish(fig, save)

    def plot_residual_boxplots(self, save=None):
        scaled = pd.DataFrame({
            self.models.label(c): self.active[c] / self.models.sigma[c] for c in AXES
        })
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.boxplot([scaled[c] for c in scaled.columns], tick_labels=scaled.columns, showfliers=True,
                   flierprops={"markersize": 2, "alpha": 0.3})
        for k, color in ((2, "orange"), (4, "red")):
            ax.axhline(k, color=color, linestyle="--", linewidth=1, label=f"{k} sigma")
        ax.set_ylabel("Residual / sigma")
        ax.set_title("Standardised residuals per axis (running readings only)")
        ax.legend()
        fig.tight_layout()
        return self.finish(fig, save)

    def plot_run_lengths(self, ks=(1, 2, 3, 4), chosen=None, limit=60, save=None):
        fig, ax = plt.subplots(figsize=(11, 5))
        grid = np.arange(0, limit + 1, 2)
        for k in ks:
            lengths = np.concatenate([self.run_lengths(c, k) for c in AXES])
            counts = [(lengths >= t).sum() for t in grid]
            ax.plot(grid, np.maximum(counts, 0.5), marker="o", markersize=3, label=f"{k} sigma")
        ax.set_yscale("log")
        ax.set_xlabel("Minimum continuous duration T (seconds)")
        ax.set_ylabel("Runs lasting at least T (all axes)")
        if chosen is not None:
            ax.axvline(chosen, color="black", linestyle="--", label=f"chosen T = {chosen:g}s")
        ax.set_title("How long normal operation stays above the regression line")
        ax.legend()
        fig.tight_layout()
        return self.finish(fig, save)

    @staticmethod
    def finish(fig, save):
        if save:
            fig.savefig(save, dpi=130, bbox_inches="tight")
        plt.show()
        plt.close(fig)
