import time

import pandas as pd

from src.database_service import DatabaseService


class StreamingSimulator:

    def __init__(self, source, db=None, table="lab1_stream"):
        if isinstance(source, pd.DataFrame):
            self.data = source.reset_index(drop=True)
        else:
            self.data = pd.read_csv(source)
        self.db = db or DatabaseService()
        self.table = table
        self.current_index = 0
        self.events = []

    def nextDataPoint(self):
        if self.current_index >= len(self.data):
            return None
        data_point = self.data.iloc[self.current_index]
        self.current_index += 1
        return data_point

    def nextBatch(self, size):
        batch = self.data.iloc[self.current_index:self.current_index + size]
        self.current_index += len(batch)
        return batch

    def send_to_database(self, data_point):
        axes = [data_point[f"Axis #{i}"] for i in range(1, 9)]
        return self.db.insert_record(data_point["Trait"], axes, data_point["Time"], self.table)

    def send_batch(self, batch):
        return self.db.insert_batch(batch, self.table)

    @staticmethod
    def as_readings(batch):
        readings = pd.DataFrame({
            "Timestamp": pd.to_datetime(batch["Time"], utc=True).dt.tz_localize(None).to_numpy()
        })
        for i in range(1, 9):
            readings[f"axis_{i}"] = batch[f"Axis #{i}"].to_numpy()
        return readings

    def record(self, closed, log):
        if closed:
            self.events.extend(closed)
            if log is not None:
                log.write(closed)

    def start_stream(self, dashboard=None, detector=None, log=None, batch_size=1,
                     delay=2.0, refresh_every=1, max_rows=None):
        print("Starting robot data stream...")
        limit = len(self.data) if max_rows is None else min(max_rows, len(self.data))
        batches = 0
        while self.current_index < limit:
            batch = self.nextBatch(min(batch_size, limit - self.current_index))
            self.send_batch(batch)
            if detector is not None:
                self.record(detector.feed_frame(self.as_readings(batch)), log)
                for item in detector.raised:
                    print(f"{item['level']} raised on {item['axis']} since {item['since']}")
            batches += 1
            if dashboard is not None and batches % refresh_every == 0:
                dashboard.show_live_dashboard()
            if self.current_index < limit:
                time.sleep(delay)
        if detector is not None:
            self.record(detector.flush(), log)
        print(f"Streaming stopped. {self.current_index} readings sent, {len(self.events)} events logged.")
        return pd.DataFrame(self.events)
