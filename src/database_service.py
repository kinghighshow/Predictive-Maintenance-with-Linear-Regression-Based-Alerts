import os

import pandas as pd
import psycopg
from psycopg import sql
from dotenv import load_dotenv

load_dotenv()

AXES = [f"axis_{i}" for i in range(1, 9)]
COLUMNS = ["id", "trait", *AXES, "year", "month", "day", "time"]
EVENT_FIELDS = [
    "level", "axis", "start_time", "end_time", "duration_s",
    "peak_deviation", "mean_deviation", "limit_c", "min_duration_s",
]


class DatabaseService:

    def __init__(self, url=None, table="lab1_training", events_table="lab1_alert_events"):
        self.url = url or os.getenv("DATABASE_URL")
        self.table = table
        self.events_table = events_table

    def connect_db(self):
        return psycopg.connect(self.url)

    def test_connection(self):
        with self.connect_db() as connection:
            version = connection.execute("SELECT version()").fetchone()[0]
        print("Database connected successfully.")
        print(version)
        return True

    def create_table(self, table=None):
        table = table or self.table
        query = sql.SQL("""
            CREATE TABLE IF NOT EXISTS {} (
                id BIGSERIAL PRIMARY KEY,
                trait VARCHAR(50),
                axis_1 DOUBLE PRECISION,
                axis_2 DOUBLE PRECISION,
                axis_3 DOUBLE PRECISION,
                axis_4 DOUBLE PRECISION,
                axis_5 DOUBLE PRECISION,
                axis_6 DOUBLE PRECISION,
                axis_7 DOUBLE PRECISION,
                axis_8 DOUBLE PRECISION,
                year INTEGER,
                month INTEGER,
                day INTEGER,
                time TIME,
                UNIQUE (year, month, day, time)
            )
        """).format(sql.Identifier(table))
        with self.connect_db() as connection:
            connection.execute(query)
        print(f"{table} table is ready.")

    def create_events_table(self):
        query = sql.SQL("""
            CREATE TABLE IF NOT EXISTS {} (
                id BIGSERIAL PRIMARY KEY,
                level VARCHAR(10),
                axis VARCHAR(10),
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                duration_s DOUBLE PRECISION,
                peak_deviation DOUBLE PRECISION,
                mean_deviation DOUBLE PRECISION,
                limit_c DOUBLE PRECISION,
                min_duration_s DOUBLE PRECISION,
                UNIQUE (level, axis, start_time)
            )
        """).format(sql.Identifier(self.events_table))
        with self.connect_db() as connection:
            connection.execute(query)
        print(f"{self.events_table} table is ready.")

    @staticmethod
    def split_timestamp(value):
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is not None:
            stamp = stamp.tz_convert("UTC").tz_localize(None)
        return stamp.year, stamp.month, stamp.day, stamp.time()

    @staticmethod
    def frame_to_rows(frame):
        stamps = pd.to_datetime(frame["Time"], utc=True).dt.tz_localize(None)
        values = frame[[f"Axis #{i}" for i in range(1, 9)]].to_numpy(dtype=float)
        return [
            (trait, *map(float, row), s.year, s.month, s.day, s.time())
            for trait, row, s in zip(frame["Trait"], values, stamps)
        ]

    def insert_query(self, table):
        marks = ", ".join(["%s"] * 13)
        return sql.SQL("""
            INSERT INTO {} (trait, axis_1, axis_2, axis_3, axis_4, axis_5,
                axis_6, axis_7, axis_8, year, month, day, time)
            VALUES ({})
            ON CONFLICT (year, month, day, time) DO NOTHING
        """).format(sql.Identifier(table), sql.SQL(marks))

    def insert_record(self, trait, axes, timestamp, table=None):
        table = table or self.table
        year, month, day, clock = self.split_timestamp(timestamp)
        row = (trait, *map(float, axes), year, month, day, clock)
        with self.connect_db() as connection:
            cursor = connection.execute(self.insert_query(table), row)
            return cursor.rowcount == 1

    def insert_batch(self, frame, table=None, chunk=2000):
        table = table or self.table
        rows = self.frame_to_rows(frame)
        query = self.insert_query(table)
        inserted = 0
        with self.connect_db() as connection:
            with connection.cursor() as cursor:
                for i in range(0, len(rows), chunk):
                    cursor.executemany(query, rows[i:i + chunk])
                    inserted += max(cursor.rowcount, 0)
        return inserted

    def load_csv(self, path, table=None):
        table = table or self.table
        frame = pd.read_csv(path)
        inserted = self.insert_batch(frame, table)
        print(f"{inserted} of {len(frame)} rows from {path} inserted into {table}.")
        return inserted

    @staticmethod
    def to_frame(records):
        frame = pd.DataFrame(records, columns=COLUMNS)
        if frame.empty:
            frame["Timestamp"] = pd.Series(dtype="datetime64[ns]")
            return frame
        date = pd.to_datetime(frame[["year", "month", "day"]])
        clock = pd.to_timedelta(frame["time"].astype(str))
        frame["Timestamp"] = date + clock
        return frame.sort_values("Timestamp").reset_index(drop=True)

    def get_records(self, table=None):
        table = table or self.table
        query = sql.SQL("SELECT * FROM {} ORDER BY id").format(sql.Identifier(table))
        with self.connect_db() as connection:
            records = connection.execute(query).fetchall()
        print(f"{len(records)} record(s) retrieved from {table}.")
        return records

    def get_frame(self, table=None):
        return self.to_frame(self.get_records(table))

    def get_recent(self, limit, table=None):
        table = table or self.table
        query = sql.SQL("SELECT * FROM {} ORDER BY id DESC LIMIT %s").format(sql.Identifier(table))
        with self.connect_db() as connection:
            records = connection.execute(query, (limit,)).fetchall()
        return self.to_frame(records[::-1])

    def count_records(self, table=None):
        table = table or self.table
        query = sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(table))
        with self.connect_db() as connection:
            return connection.execute(query).fetchone()[0]

    def clear_table(self, table=None):
        table = table or self.table
        query = sql.SQL("TRUNCATE TABLE {}").format(sql.Identifier(table))
        with self.connect_db() as connection:
            connection.execute(query)
        print(f"{table} cleared.")

    def insert_events(self, events):
        query = sql.SQL("""
            INSERT INTO {} (level, axis, start_time, end_time, duration_s,
                peak_deviation, mean_deviation, limit_c, min_duration_s)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (level, axis, start_time) DO NOTHING
        """).format(sql.Identifier(self.events_table))
        rows = [
            (r.level, r.axis, r.start.to_pydatetime(), r.end.to_pydatetime(),
             float(r.duration_s), float(r.peak_deviation), float(r.mean_deviation),
             float(r.limit_c), float(r.min_duration_s))
            for r in events.itertuples()
        ]
        with self.connect_db() as connection:
            with connection.cursor() as cursor:
                cursor.executemany(query, rows)
        return len(rows)

    def get_events(self):
        query = sql.SQL("SELECT {} FROM {} ORDER BY start_time, axis").format(
            sql.SQL(", ").join(sql.Identifier(f) for f in EVENT_FIELDS),
            sql.Identifier(self.events_table),
        )
        with self.connect_db() as connection:
            records = connection.execute(query).fetchall()
        frame = pd.DataFrame(records, columns=EVENT_FIELDS)
        return frame.rename(columns={"start_time": "start", "end_time": "end"})

    def clear_events(self):
        query = sql.SQL("TRUNCATE TABLE {}").format(sql.Identifier(self.events_table))
        with self.connect_db() as connection:
            connection.execute(query)
