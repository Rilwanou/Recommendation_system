from pathlib import Path
import duckdb

from data_preprocessing.merge import generate_final_table_csv

def test_merge_generates_readable_csv(tmp_path: Path) -> None:
    sql_path = Path("back-end/src/data_preprocessing/merge.sql")
    out_csv = tmp_path / "final_table.csv"

    generated = generate_final_table_csv(sql_path=sql_path, out_csv=out_csv, table_name="final_table")

    assert generated.exists()
    assert generated.stat().st_size > 0

    with duckdb.connect(":memory:") as con:
        n = con.execute(f"""
            SELECT COUNT(*) FROM read_csv('{generated.as_posix()}', header=true, encoding='utf-8')
        """).fetchone()[0]
        assert n >= 0
