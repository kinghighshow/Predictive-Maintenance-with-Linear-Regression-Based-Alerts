import numpy as np
import pandas as pd

from src.database_service import AXES


class SyntheticDataGenerator:

    def __init__(self, training, seed=8010):
        self.training = training.reset_index(drop=True)
        self.rng = np.random.default_rng(seed)
        self.profile = self.describe(self.training)
        gaps = self.training["Timestamp"].diff().dt.total_seconds().dropna()
        self.intervals = gaps[(gaps >= 1.5) & (gaps <= 3.0)].to_numpy()
        self.faults = pd.DataFrame()

    @staticmethod
    def describe(frame):
        stats = frame[AXES].agg(["mean", "std", "min", "max"]).T
        stats["zero_share"] = (frame[AXES] == 0).mean()
        return stats

    @staticmethod
    def normalize(values, low, high):
        return (values - low) / (high - low)

    @staticmethod
    def denormalize(values, low, high):
        return values * (high - low) + low

    @staticmethod
    def standardize(values, mean, std):
        return (values - mean) / (std if std else 1.0)

    @staticmethod
    def destandardize(values, mean, std):
        return values * std + mean

    def resample(self, rows, block=1800):
        source = self.training[AXES].to_numpy(dtype=float)
        pieces, total = [], 0
        while total < rows:
            start = self.rng.integers(0, len(source) - block)
            pieces.append(source[start:start + block])
            total += block
        return np.vstack(pieces)[:rows]

    def jitter(self, values, spread=0.03):
        factor = 1 + self.rng.normal(0, spread, values.shape)
        return np.where(values > 0, values * factor, 0.0)

    def align(self, values):
        aligned = np.empty_like(values)
        for j, column in enumerate(AXES):
            target = self.profile.loc[column]
            z = self.standardize(values[:, j], values[:, j].mean(), values[:, j].std())
            scaled = self.destandardize(z, target["mean"], target["std"])
            unit = self.normalize(scaled, target["min"], target["max"])
            aligned[:, j] = self.denormalize(np.clip(unit, 0, 1), target["min"], target["max"])
        return aligned

    @staticmethod
    def default_plan():
        return [
            {"axis": 2, "at_hours": 5.0, "duration_s": 90, "sigma": 3.0, "kind": "step"},
            {"axis": 5, "at_hours": 9.0, "duration_s": 20, "sigma": 5.0, "kind": "step"},
            {"axis": 3, "at_hours": 20.0, "duration_s": 240, "sigma": 5.0, "kind": "step"},
            {"axis": 7, "at_hours": 27.0, "duration_s": 120, "sigma": 2.6, "kind": "step"},
            {"axis": 6, "at_hours": 33.0, "duration_s": 1500, "sigma": 6.0, "kind": "ramp"},
            {"axis": 1, "at_hours": 40.0, "duration_s": 300, "sigma": 1.0, "kind": "step"},
            {"axis": 2, "at_hours": 44.0, "duration_s": 150, "sigma": 4.5, "kind": "step"},
            {"axis": 3, "at_hours": 44.0, "duration_s": 150, "sigma": 4.5, "kind": "step"},
            {"axis": 4, "at_hours": 44.0, "duration_s": 150, "sigma": 4.5, "kind": "step"},
        ]

    def inject(self, values, offsets, origin, plan, models):
        records = []
        for fault in plan:
            start = fault["at_hours"] * 3600
            low, high = np.searchsorted(offsets, [start, start + fault["duration_s"]])
            if high - low < 2 or high > len(offsets):
                continue
            column = AXES[fault["axis"] - 1]
            line = models.predict([origin + pd.Timedelta(seconds=start)], column)[0]
            peak = line + fault["sigma"] * models.sigma[column]
            lift = peak if fault["kind"] == "step" else np.linspace(0, peak, high - low)
            values[low:high, fault["axis"] - 1] += lift
            records.append({
                "axis": f"Axis #{fault['axis']}",
                "start": (origin + pd.Timedelta(seconds=offsets[low])).round("ms"),
                "end": (origin + pd.Timedelta(seconds=offsets[high - 1])).round("ms"),
                "length_s": round(float(offsets[high - 1] - offsets[low]), 1),
                "sigma": fault["sigma"],
                "kind": fault["kind"],
            })
        return pd.DataFrame(records)

    def generate(self, start, hours, models, plan=None):
        span = hours * 3600
        steps = self.rng.choice(self.intervals, int(span / self.intervals.mean()) + 200)
        offsets = np.round(np.cumsum(steps), 3)
        offsets = offsets[offsets <= span]
        origin = pd.Timestamp(start)
        values = self.align(self.jitter(self.resample(len(offsets))))
        self.faults = self.inject(values, offsets, origin, plan or self.default_plan(), models)
        stamps = pd.Series((origin + pd.to_timedelta(offsets, unit="s")).round("ms"))
        frame = pd.DataFrame({"Trait": "current"}, index=range(len(offsets)))
        for i in range(1, 9):
            frame[f"Axis #{i}"] = values[:, i - 1].round(3)
        frame["Time"] = stamps.dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3] + "Z"
        return frame

    @staticmethod
    def readings(frame):
        readings = frame[[f"Axis #{i}" for i in range(1, 9)]].astype(float)
        readings.columns = AXES
        return readings

    @staticmethod
    def running(readings):
        return readings[readings.sum(axis=1) > 0]

    def idle_share(self, frame):
        return {
            "train": float(1 - (self.training[AXES].sum(axis=1) > 0).mean()),
            "test": float(1 - (self.readings(frame).sum(axis=1) > 0).mean()),
        }

    def compare(self, frame):
        test = self.running(self.readings(frame))
        train = self.running(self.training[AXES])
        table = pd.DataFrame({
            "train_mean": train.mean(),
            "test_mean": test.mean(),
            "train_std": train.std(),
            "test_std": test.std(),
            "train_max": train.max(),
            "test_max": test.max(),
        })
        table.index = [f"Axis #{i}" for i in range(1, 9)]
        return table
