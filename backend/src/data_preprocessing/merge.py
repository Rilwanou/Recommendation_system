from __future__ import annotations

from pathlib import Path
import duckdb


def generate_final_table_csv(
    sql_path: Path, out_csv: Path, table_name: str = "final_table"
) -> Path:
    """
    Execute the merge SQL in an in-memory DuckDB and export the resulting table to CSV.
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with duckdb.connect(":memory:") as con:
        con.execute(sql_path.read_text(encoding="utf-8"))
        con.execute(f"""
            COPY (SELECT * FROM {table_name})
            TO '{out_csv.as_posix()}'
            (FORMAT CSV, HEADER, DELIMITER ',');
        """)

    return out_csv


def main() -> None:
    sql_path = Path("back-end/src/data_preprocessing/merge.sql")
    out_csv = Path("data/processed/final_table.csv")

    generated = generate_final_table_csv(
        sql_path=sql_path, out_csv=out_csv, table_name="final_table"
    )
    print(f"✅ CSV généré: {generated}")


if __name__ == "__main__":
    main()
