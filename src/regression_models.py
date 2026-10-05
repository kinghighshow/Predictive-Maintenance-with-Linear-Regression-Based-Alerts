import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.database_service import AXES


class AxisRegression:

    def __init__(self, columns=None):
        self.columns = columns or AXES
        self.models = {}
        self.sigma = {}
        self.origin = None
        self.table = None
        self.running_only = True
        self.idle_share = 0.0

    @staticmethod
    def running_mask(frame, columns=None):
        return frame[columns or AXES].astype(float).sum(axis=1) > 0

    @staticmethod
    def label(column):
        return f"Axis #{column.split('_')[1]}"

    def seconds(self, timestamps):
        delta = pd.to_datetime(timestamps) - self.origin
        return np.asarray(delta / pd.Timedelta(seconds=1), dtype=float)

    def fit(self, frame, running_only=True):
        self.origin = frame["Timestamp"].min()
        self.running_only = running_only
        self.idle_share = float(1 - self.running_mask(frame, self.columns).mean())
        if running_only:
            frame = frame[self.running_mask(frame, self.columns)]
        x = self.seconds(frame["Timestamp"]).reshape(-1, 1)
        rows = []
        for column in self.columns:
            y = frame[column].to_numpy(dtype=float)
            model = LinearRegression().fit(x, y)
            residual = y - model.predict(x)
            self.models[column] = model
            self.sigma[column] = float(residual.std())
            rows.append({
                "axis": self.label(column),
                "slope_per_second": model.coef_[0],
                "slope_per_hour": model.coef_[0] * 3600,
                "intercept": model.intercept_,
                "r2": model.score(x, y),
                "resid_std": residual.std(),
                "resid_p99": np.percentile(residual, 99),
                "resid_max": residual.max(),
                "samples": len(y),
            })
        self.table = pd.DataFrame(rows).set_index("axis")
        return self

    def summary(self):
        return self.table

    def predict(self, timestamps, column):
        x = self.seconds(timestamps).reshape(-1, 1)
        return self.models[column].predict(x)

    def predicted(self, frame):
        out = {column: self.predict(frame["Timestamp"], column) for column in self.columns}
        return pd.DataFrame(out, index=frame.index)

    def residuals(self, frame):
        return frame[self.columns].astype(float) - self.predicted(frame)
