import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL


class EnergyDatabase:
    def __init__(self):
        self.sql_dir = Path(__file__).parent / "sql"

        env_path = Path(__file__).resolve().parents[1] / ".env"
        load_dotenv(env_path)

        db_url = URL.create(
            drivername="postgresql+psycopg2",
            username=os.environ.get("POSTGRES_USER"),
            password=os.environ.get("POSTGRES_PASSWORD"),
            host=os.environ.get("POSTGRES_HOST"),
            port=os.environ.get("POSTGRES_PORT"),
            database=os.environ.get("POSTGRES_DB"),
        )
        self.engine = create_engine(db_url)

    def _read_sql_file(self, query_name: str) -> str:
        """
        Pobiera zawartość pliku SQL na podstawie nazwy
        """
        file_path = self.sql_dir / f"{query_name}.sql"
        if not file_path.exists():
            raise FileNotFoundError(f"Nie znaleziono zapytania: {file_path}")

        with open(file_path, encoding="utf-8") as file:
            return file.read()

    def fetch_by_datetime(self, query_name: str, columns: tuple = None, **params) -> pd.DataFrame:
        query_text = self._read_sql_file(query_name)

        cols_str = ", ".join(columns) if columns else "*"
        query_text = query_text.replace("{columns}", cols_str)

        where_conditions = []
        if "start_date" in params:
            where_conditions.append("datetime_utc >= %(start_date)s")
        if "end_date" in params:
            where_conditions.append("datetime_utc < %(end_date)s")

        if where_conditions:
            where_clause = "WHERE " + " AND ".join(where_conditions)
        else:
            where_clause = ""

        query_text = query_text.replace("{where_filters}", where_clause)

        return pd.read_sql_query(query_text, con=self.engine, params=params)
