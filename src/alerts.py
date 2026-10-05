import os

import numpy as np
import pandas as pd

EVENT_COLUMNS = [
    "level", "axis", "start", "end", "duration_s",
    "peak_deviation", "mean_deviation", "limit_c", "min_duration_s",
]
LEVELS = ("ALERT", "ERROR")


class AlertDetector:

    def __init__(self, models, min_sigma=2.0, max_sigma=4.0, duration=30.0, max_gap=10.0):
        self.models = models
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.duration = duration
        self.max_gap = max_gap
        self.min_c = {c: min_sigma * models.sigma[c] for c in models.columns}
        self.max_c = {c: max_sigma * models.sigma[c] for c in models.columns}
        self.reset()

    def reset(self):
        self.runs = {(c, level): None for c in self.models.columns for level in LEVELS}
        self.raised = []

    def limit(self, level, column):
        return self.min_c[column] if level == "ALERT" else self.max_c[column]

    def stamp(self, seconds):
        return (self.models.origin + pd.to_timedelta(seconds, unit="s")).round("ms")

    def close_run(self, column, level, run):
        length = run["last"] - run["start"]
        if length < self.duration:
            return None
        return {
            "level": level,
            "axis": self.models.label(column),
            "start": self.stamp(run["start"]),
            "end": self.stamp(run["last"]),
            "duration_s": round(length, 1),
            "peak_deviation": round(run["peak"], 3),
            "mean_deviation": round(run["total"] / run["count"], 3),
            "limit_c": round(self.limit(level, column), 3),
            "min_duration_s": self.duration,
        }

    def feed_frame(self, frame):
        self.raised = []
        closed = []
        seconds = self.models.seconds(frame["Timestamp"]).tolist()
        residuals = self.models.residuals(frame)
        for column in self.models.columns:
            deviations = residuals[column].tolist()
            for level in LEVELS:
                limit = self.limit(level, column)
                run = self.runs[(column, level)]
                for t, d in zip(seconds, deviations):
                    if run is not None and t - run["last"] > self.max_gap:
                        event = self.close_run(column, level, run)
                        if event:
                            closed.append(event)
                        run = None
                    if d >= limit:
                        if run is None:
                            run = {"start": t, "last": t, "peak": d, "total": d,
                                   "count": 1, "raised": False}
                        else:
                            run["last"] = t
                            run["peak"] = max(run["peak"], d)
                            run["total"] += d
                            run["count"] += 1
                        if not run["raised"] and t - run["start"] >= self.duration:
                            run["raised"] = True
                            self.raised.append({
                                "level": level,
                                "axis": self.models.label(column),
                                "since": self.stamp(run["start"]),
                            })
                    elif run is not None:
                        event = self.close_run(column, level, run)
                        if event:
                            closed.append(event)
                        run = None
                self.runs[(column, level)] = run
        return closed

    def flush(self):
        closed = []
        for (column, level), run in self.runs.items():
            if run is not None:
                event = self.close_run(column, level, run)
                if event:
                    closed.append(event)
                self.runs[(column, level)] = None
        return closed

    def active(self):
        found = []
        for (column, level), run in self.runs.items():
            if run is not None and run["last"] - run["start"] >= self.duration:
                found.append({
                    "level": level,
                    "axis": self.models.label(column),
                    "since": self.stamp(run["start"]),
                })
        return found

    @staticmethod
    def as_frame(events):
        frame = pd.DataFrame(events, columns=EVENT_COLUMNS)
        return frame.sort_values(["start", "axis", "level"]).reset_index(drop=True)

    def detect(self, frame):
        self.reset()
        events = self.feed_frame(frame) + self.flush()
        return self.as_frame(events)

    def expected_level(self, sigma, length):
        if length < self.duration:
            return "NONE"
        if sigma >= self.max_sigma:
            return "ERROR"
        if sigma >= self.min_sigma:
            return "ALERT"
        return "NONE"

    def evaluate(self, events, faults, tolerance=10.0):
        pad = pd.Timedelta(seconds=tolerance)
        rows = []
        claimed = pd.Series(False, index=events.index)
        for fault in faults.itertuples():
            hit = events[
                (events["axis"] == fault.axis)
                & (events["start"] <= fault.end + pad)
                & (events["end"] >= fault.start - pad)
            ]
            claimed[hit.index] = True
            levels = sorted(hit["level"].unique())
            first = hit["start"].min() if len(hit) else pd.NaT
            rows.append({
                "axis": fault.axis,
                "injected_at": fault.start,
                "length_s": fault.length_s,
                "sigma": fault.sigma,
                "kind": fault.kind,
                "expected": self.expected_level(fault.sigma, fault.length_s),
                "detected": "+".join(levels) if levels else "NONE",
                "delay_s": (first - fault.start).total_seconds() if len(hit) else np.nan,
            })
        return pd.DataFrame(rows), events[~claimed]


class EventLog:

    def __init__(self, path="data/alert_log.csv", db=None):
        self.path = path
        self.db = db

    def reset(self):
        if os.path.exists(self.path):
            os.remove(self.path)
        if self.db is not None:
            self.db.clear_events()

    def write(self, events):
        frame = pd.DataFrame(events, columns=EVENT_COLUMNS)
        if frame.empty:
            return 0
        frame.to_csv(self.path, mode="a", header=not os.path.exists(self.path), index=False)
        if self.db is not None:
            self.db.insert_events(frame)
        return len(frame)

    def read(self):
        if not os.path.exists(self.path):
            return pd.DataFrame(columns=EVENT_COLUMNS)
        frame = pd.read_csv(self.path, parse_dates=["start", "end"])
        return frame.sort_values(["start", "axis", "level"]).reset_index(drop=True)
