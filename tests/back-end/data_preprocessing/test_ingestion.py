from __future__ import annotations

from pathlib import Path
import types

import pandas as pd
import pytest
from pydantic import ValidationError
from data_ingestion import ingestion as mod



def _write_csv(path: Path, df: pd.DataFrame) -> None:
    """Write a DataFrame to CSV (utf-8) without index."""
    path.write_text(df.to_csv(index=False), encoding="utf-8")


def _sample_df() -> pd.DataFrame:
    """Small sample DataFrame for deterministic tests."""
    return pd.DataFrame({"id": [1, 2], "name": ["alice", "bob"]})


class TestCsvSourceValidation:
    def test_requires_exactly_one_of_url_or_path(self) -> None:
        with pytest.raises(ValueError, match="exactly one of `url` or `path`"):
            mod.CsvSource()  

    def test_rejects_both_url_and_path(self, tmp_path: Path) -> None:
        local = tmp_path / "file.csv"
        local.write_text("a,b\n1,2\n", encoding="utf-8")

        with pytest.raises(ValueError, match="exactly one of `url` or `path`"):
            mod.CsvSource(url="https://example.com/file.csv", path=local) 

    def test_rejects_missing_local_file(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.csv"
        with pytest.raises(FileNotFoundError, match="CSV file not found"):
            mod.CsvSource(path=missing)

    def test_accepts_existing_local_file(self, tmp_path: Path) -> None:
        local = tmp_path / "ok.csv"
        local.write_text("a,b\n1,2\n", encoding="utf-8")

        src = mod.CsvSource(path=local)
        assert src.path == local
        assert src.url is None

    def test_accepts_http_url(self) -> None:
        src = mod.CsvSource(url="https://example.com/data.csv") 
        assert src.url is not None
        assert src.path is None

class TestSaveCsvParamsValidation:
    def test_rejects_empty_filename(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError):
            mod.SaveCsvParams(raw_dir=tmp_path, filename="")

    def test_accepts_valid_params(self, tmp_path: Path) -> None:
        params = mod.SaveCsvParams(raw_dir=tmp_path, filename="out.csv")
        assert params.raw_dir == tmp_path
        assert params.filename == "out.csv"

class TestSafeLen:
    def test_none_returns_0(self) -> None:
        assert mod.safe_len(None) == 0

    def test_dataframe_returns_length(self) -> None:
        df = pd.DataFrame({"x": [10, 20, 30]})
        assert mod.safe_len(df) == 3

class TestLoadCsv:
    def test_loads_from_local_path(self, tmp_path: Path) -> None:
        df_in = _sample_df()
        csv_path = tmp_path / "in.csv"
        _write_csv(csv_path, df_in)

        df_out = mod.load_csv(mod.CsvSource(path=csv_path))
        pd.testing.assert_frame_equal(df_out, df_in)

    def test_loads_from_url_by_calling_pandas_read_csv(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        We don't want real HTTP calls in unit tests.
        So we mock pandas.read_csv and verify it is called with the URL string.
        """
        expected = _sample_df()

        def fake_read_csv(arg, *args, **kwargs):
            assert isinstance(arg, str)
            assert arg.startswith("https://")
            return expected

        monkeypatch.setattr(pd, "read_csv", fake_read_csv)

        df_out = mod.load_csv(mod.CsvSource(url="https://example.com/data.csv"))  
        pd.testing.assert_frame_equal(df_out, expected)

class TestSaveCsv:
    def test_skips_when_df_is_none(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        mod.save_csv(None, "out.csv", tmp_path)

        captured = capsys.readouterr().out
        assert "Skipped out.csv" in captured
        assert not (tmp_path / "out.csv").exists()

    def test_writes_csv_when_df_exists(self, tmp_path: Path) -> None:
        df = _sample_df()
        mod.save_csv(df, "out.csv", tmp_path)

        out_path = tmp_path / "out.csv"
        assert out_path.exists()

        df_back = pd.read_csv(out_path)
        pd.testing.assert_frame_equal(df_back, df)

    def test_creates_directory_if_missing(self, tmp_path: Path) -> None:
        df = _sample_df()
        target_dir = tmp_path / "nested" / "raw"

        assert not target_dir.exists()
        mod.save_csv(df, "out.csv", target_dir)
        assert target_dir.exists()
        assert (target_dir / "out.csv").exists()


class TestMiniFlow:
    def test_load_url_then_save(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """
        This mimics your real flow:
        - load_csv from URL (but mocked)
        - save_csv to raw_dir
        """
        expected = _sample_df()

        monkeypatch.setattr(pd, "read_csv", lambda *_a, **_kw: expected)

        df = mod.load_csv(mod.CsvSource(url="https://example.com/customers.csv"))  # type: ignore[arg-type]
        mod.save_csv(df, "customers.csv", tmp_path)

        assert (tmp_path / "customers.csv").exists()
        df_back = pd.read_csv(tmp_path / "customers.csv")
        pd.testing.assert_frame_equal(df_back, expected)
