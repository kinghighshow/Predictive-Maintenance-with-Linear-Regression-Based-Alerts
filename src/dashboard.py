import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import clear_output, display
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.database_service import AXES

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


class RobotDashboard:

    def __init__(self, db, models, detector, log, table="lab1_stream", max_points=2500):
        self.db = db
        self.models = models
        self.detector = detector
        self.log = log
        self.table = table
        self.max_points = max_points
        self.palette = plt.cm.tab10.colors[:8]
        self.data = pd.DataFrame()

    def load(self):
        self.data = self.db.get_frame(self.table)
        return self.data

    @staticmethod
    def apply_filters(frame, month=None, day=None, hour=None):
        stamp = frame["Timestamp"]
        mask = pd.Series(True, index=frame.index)
        if month is not None:
            mask &= stamp.dt.month == month
        if day is not None:
            mask &= stamp.dt.day == day
        if hour is not None:
            mask &= stamp.dt.hour == hour
        return frame[mask]

    def choices(self, month=None, day=None):
        scope = self.apply_filters(self.data, month=month)
        months = self.data["Timestamp"].dt.month.unique()
        days = scope["Timestamp"].dt.day.unique()
        hours = self.apply_filters(scope, day=day)["Timestamp"].dt.hour.unique()
        return [sorted(int(v) for v in values) for values in (months, days, hours)]

    @staticmethod
    def window_label(month, day, hour):
        parts = []
        if month is not None:
            parts.append(MONTHS[month - 1])
        if day is not None:
            parts.append(f"day {day:02d}")
        if hour is not None:
            parts.append(f"{hour:02d}:00-{hour:02d}:59")
        return " / ".join(parts) if parts else "All data"

    def events_between(self, start, end):
        events = self.log.read()
        if events.empty:
            return events
        return events[(events["end"] >= start) & (events["start"] <= end)]

    def downsample(self, frame):
        frame = frame.reset_index(drop=True)
        if len(frame) <= self.max_points:
            return frame, 1
        step = int(np.ceil(len(frame) / self.max_points))
        grouped = frame.groupby(np.arange(len(frame)) // step)
        reduced = grouped[AXES].max()
        reduced.insert(0, "Timestamp", grouped["Timestamp"].first())
        return reduced.reset_index(drop=True), step

    def draw_panel(self, ax, frame, column, events, compact):
        number = int(column.split("_")[1])
        stamps = frame["Timestamp"]
        line = self.models.predict(stamps, column)
        ax.plot(stamps, frame[column], color=self.palette[number - 1], linewidth=0.8)
        ax.plot(stamps, line, color="black", linestyle="--", linewidth=1)
        ax.plot(stamps, line + self.detector.min_c[column], color="orange", linestyle=":", linewidth=1.2)
        ax.plot(stamps, line + self.detector.max_c[column], color="red", linestyle=":", linewidth=1.2)
        ax.set_title(self.models.label(column), fontsize=10)
        top = ax.get_ylim()[1]
        mine = events[events["axis"] == self.models.label(column)].sort_values("level")
        for row in mine.itertuples():
            color = "red" if row.level == "ERROR" else "orange"
            middle = row.start + (row.end - row.start) / 2
            ax.axvspan(row.start, row.end, color=color, alpha=0.3)
            height = 0.93 if row.level == "ERROR" else 0.80
            ax.plot(middle, top * height, marker="v", color=color, markersize=6)
            ax.annotate(f"{row.duration_s:.0f}s", (middle, top * height), xytext=(0, 5),
                        textcoords="offset points", ha="center", fontsize=6.5 if compact else 8)
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    def legend_handles(self):
        return [
            Line2D([0], [0], color="gray", label="Current (peak per point when zoomed out)"),
            Line2D([0], [0], color="black", linestyle="--", label="Regression line"),
            Line2D([0], [0], color="orange", linestyle=":", label="Line + MinC"),
            Line2D([0], [0], color="red", linestyle=":", label="Line + MaxC"),
            Patch(color="orange", alpha=0.4, label="Alert"),
            Patch(color="red", alpha=0.4, label="Error"),
        ]

    def draw(self, month=None, day=None, hour=None, axis=None):
        frame = self.apply_filters(self.data, month, day, hour)
        if frame.empty:
            print("No readings for this selection.")
            return None, frame, frame
        events = self.events_between(frame["Timestamp"].min(), frame["Timestamp"].max())
        if axis is not None and len(events):
            events = events[events["axis"] == f"Axis #{axis}"]
        reduced, _ = self.downsample(frame)
        if axis is None:
            fig, grid = plt.subplots(4, 2, figsize=(15, 12), sharex=True)
            panels = list(zip(grid.ravel(), AXES))
        else:
            fig, single = plt.subplots(figsize=(14, 5.5))
            panels = [(single, f"axis_{axis}")]
        for ax, column in panels:
            self.draw_panel(ax, reduced, column, events, compact=axis is None)
        alerts = int((events["level"] == "ALERT").sum()) if len(events) else 0
        errors = int((events["level"] == "ERROR").sum()) if len(events) else 0
        running = self.models.running_mask(frame).mean() * 100
        fig.suptitle(f"{self.window_label(month, day, hour)}  |  {len(frame):,} readings ({running:.0f}% running)  |  "
                     f"alerts: {alerts}  |  errors: {errors}", fontsize=13)
        fig.legend(handles=self.legend_handles(), loc="lower center", ncol=6, fontsize=9)
        fig.tight_layout(rect=[0, 0.04, 1, 0.96])
        return fig, frame, events

    def show(self, month=None, day=None, hour=None, axis=None):
        fig, frame, events = self.draw(month, day, hour, axis)
        if fig is None:
            return
        plt.show()
        plt.close(fig)
        if len(events):
            display(events[["level", "axis", "start", "end", "duration_s", "peak_deviation"]]
                    .reset_index(drop=True))
        else:
            print("No alerts or errors in this window.")

    def save_view(self, path, month=None, day=None, hour=None, axis=None):
        fig, _, _ = self.draw(month, day, hour, axis)
        if fig is not None:
            fig.savefig(path, dpi=130, bbox_inches="tight")
            plt.close(fig)

    def interactive(self):
        import ipywidgets as widgets

        self.load()
        months, days, hours = self.choices()
        month = widgets.Dropdown(description="Month", options=[("All", None)] + [(MONTHS[m - 1], m) for m in months])
        day = widgets.Dropdown(description="Day", options=[("All", None)] + [(f"{d:02d}", d) for d in days])
        hour = widgets.Dropdown(description="Hour", options=[("All", None)] + [(f"{h:02d}:00", h) for h in hours])
        axis = widgets.Dropdown(description="Axis", options=[("All axes", None)] + [(f"Axis #{i}", i) for i in range(1, 9)])

        def reset(control, values, label):
            kept = control.value
            control.options = [("All", None)] + [(label(v), v) for v in values]
            control.value = kept if kept in values else None

        def on_month(change):
            _, new_days, new_hours = self.choices(month.value, None)
            reset(day, new_days, lambda d: f"{d:02d}")
            reset(hour, new_hours, lambda h: f"{h:02d}:00")

        def on_day(change):
            _, _, new_hours = self.choices(month.value, day.value)
            reset(hour, new_hours, lambda h: f"{h:02d}:00")

        month.observe(on_month, names="value")
        day.observe(on_day, names="value")
        output = widgets.interactive_output(
            self.show, {"month": month, "day": day, "hour": hour, "axis": axis}
        )
        display(widgets.VBox([widgets.HBox([month, day, hour, axis]), output]))

    def show_live_dashboard(self, window=120):
        recent = self.db.get_recent(int(window / 1.8) + 30, self.table)
        if recent.empty:
            print("No data available.")
            return
        recent = recent[recent["Timestamp"] >= recent["Timestamp"].max() - pd.Timedelta(seconds=window)]
        events = self.events_between(recent["Timestamp"].min(), recent["Timestamp"].max())
        active = self.detector.active()
        clear_output(wait=True)
        fig, ax = plt.subplots(figsize=(12, 6))
        for number, column in enumerate(AXES, 1):
            color = self.palette[number - 1]
            ax.plot(recent["Timestamp"], recent[column], color=color, label=f"Axis #{number}")
            ax.plot(recent["Timestamp"], self.models.predict(recent["Timestamp"], column),
                    color=color, linestyle="--", linewidth=0.8, alpha=0.7)
        for row in events.sort_values("level").itertuples():
            ax.axvspan(row.start, row.end, color="red" if row.level == "ERROR" else "orange", alpha=0.2)
        if any(a["level"] == "ERROR" for a in active):
            status = "ERROR: " + ", ".join(sorted({a["axis"] for a in active if a["level"] == "ERROR"}))
        elif active:
            status = "ALERT: " + ", ".join(sorted({a["axis"] for a in active}))
        else:
            status = "NORMAL"
        ax.set_xlabel("Time")
        ax.set_ylabel("Axis reading")
        ax.set_title(f"Real-Time Robot Dashboard  |  status {status}")
        ax.legend(loc="upper left", ncol=4, fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        plt.close(fig)
