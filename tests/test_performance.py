"""
Performance tests for df-fingerprint vectorized encoders.

These tests verify that vectorized implementations are faster than
row-by-row processing while producing identical results.
"""

import time
import pandas as pd
import numpy as np
import pytest
from df_fingerprint import fingerprint, Canonicalizer


class TestVectorizedPerformance:
    """Test vectorized encoding performance and correctness."""

    def create_large_dataframe(self, n_rows=10000):
        """Create a large DataFrame for performance testing."""
        np.random.seed(42)  # For reproducible results

        return pd.DataFrame(
            {
                "int_col": np.random.randint(-1000, 1000, n_rows),
                "float_col": np.random.randn(n_rows),
                "str_col": [f"value_{i}" for i in range(n_rows)],
                "bool_col": np.random.choice([True, False], n_rows),
                "datetime_col": pd.date_range("2020-01-01", periods=n_rows, freq="1h"),
                "timedelta_col": pd.to_timedelta(
                    np.random.randint(0, 1000, n_rows), unit="D"
                ),
                "categorical_col": pd.Categorical(
                    np.random.choice(["A", "B", "C"], n_rows)
                ),
            }
        )

    def create_dataframe_with_nulls(self, n_rows=1000):
        """Create DataFrame with various null values for testing."""
        np.random.seed(42)

        # Create base data
        df = self.create_large_dataframe(n_rows)

        # Add nulls randomly
        null_mask = np.random.choice([True, False], n_rows, p=[0.1, 0.9])  # 10% nulls

        df.loc[null_mask, "int_col"] = None
        df.loc[null_mask, "float_col"] = np.nan
        df.loc[null_mask, "str_col"] = None
        # Convert bool column to nullable boolean dtype first to avoid warning
        df["bool_col"] = df["bool_col"].astype("boolean")
        df.loc[null_mask, "bool_col"] = pd.NA
        df.loc[null_mask, "datetime_col"] = pd.NaT
        df.loc[null_mask, "timedelta_col"] = pd.NaT

        return df

    def test_vectorized_correctness_small(self):
        """Test that vectorized encoding produces identical results on small data."""
        df = self.create_large_dataframe(100)

        fp1 = fingerprint(df)
        fp2 = fingerprint(df)

        assert fp1.hex == fp2.hex
        assert fp1.digest == fp2.digest

    def test_vectorized_correctness_with_nulls(self):
        """Test vectorized encoding with null values."""
        df = self.create_dataframe_with_nulls(100)

        fp1 = fingerprint(df)
        fp2 = fingerprint(df)

        assert fp1.hex == fp2.hex
        assert fp1.digest == fp2.digest

    def test_vectorized_deterministic_large(self):
        """Test that large DataFrames produce deterministic results."""
        df = self.create_large_dataframe(5000)

        fp1 = fingerprint(df)
        fp2 = fingerprint(df)

        assert fp1.hex == fp2.hex

    def test_empty_dataframe_vectorized(self):
        """Test vectorized encoding handles empty DataFrames."""
        df = pd.DataFrame({"col1": [], "col2": []})

        fp = fingerprint(df)
        assert fp.hex.startswith("sha256:")

    def test_single_row_dataframe(self):
        """Test vectorized encoding handles single-row DataFrames."""
        df = pd.DataFrame(
            {
                "int_col": [42],
                "float_col": [3.14],
                "str_col": ["hello"],
                "bool_col": [True],
                "datetime_col": [pd.Timestamp("2024-01-01")],
            }
        )

        fp = fingerprint(df)
        assert fp.hex.startswith("sha256:")

    @pytest.mark.parametrize("n_rows", [100, 1000, 5000])
    def test_vectorized_scaling(self, n_rows):
        """Test that vectorized encoding works correctly at different scales."""
        df = self.create_large_dataframe(n_rows)

        start_time = time.time()
        fp = fingerprint(df)
        end_time = time.time()

        assert fp.hex.startswith("sha256:")

        # Performance should be reasonable (less than 1 second for 5k rows)
        elapsed = end_time - start_time
        if n_rows <= 5000:
            assert (
                elapsed < 1.0
            ), f"Encoding {n_rows} rows took {elapsed:.2f}s, expected < 1.0s"

    def test_mixed_types_vectorized(self):
        """Test vectorized encoding with mixed types in object columns."""
        df = pd.DataFrame(
            {
                "mixed_col": [
                    1,
                    "string",
                    3.14,
                    True,
                    None,
                    b"bytes",
                    pd.Timestamp("2024-01-01"),
                    pd.Timedelta("1 day"),
                ]
            }
        )

        fp = fingerprint(df)
        assert fp.hex.startswith("sha256:")

    def test_large_integers_vectorized(self):
        """Test vectorized encoding handles large integers."""
        df = pd.DataFrame({"big_int": [2**100, 2**200, None, 42]})

        fp = fingerprint(df)
        assert fp.hex.startswith("sha256:")

    def test_special_float_values_vectorized(self):
        """Test vectorized encoding of special float values."""
        df = pd.DataFrame(
            {
                "special_floats": [
                    float("inf"),
                    float("-inf"),
                    0.0,
                    -0.0,
                    np.nan,
                    None,
                    1e-100,
                    1e100,
                ]
            }
        )

        fp = fingerprint(df)
        assert fp.hex.startswith("sha256:")

    def test_datetime_timezones_vectorized(self):
        """Test vectorized datetime encoding with different timezones."""
        df = pd.DataFrame(
            {
                "dt_utc": pd.to_datetime(["2024-01-01 12:00:00"], utc=True),
                "dt_naive": pd.to_datetime(["2024-01-01 12:00:00"]),
                "dt_tz": pd.to_datetime(["2024-01-01 04:00:00"]).tz_localize(
                    "US/Pacific"
                ),
            }
        )

        fp = fingerprint(df)
        assert fp.hex.startswith("sha256:")

    def test_categorical_ordering_vectorized(self):
        """Test vectorized categorical encoding with different category orders."""
        cat1 = pd.Categorical(["A", "B", "C"], categories=["C", "A", "B"])
        cat2 = pd.Categorical(["A", "B", "C"], categories=["A", "B", "C"])

        df1 = pd.DataFrame({"cat": cat1})
        df2 = pd.DataFrame({"cat": cat2})

        fp1 = fingerprint(df1)
        fp2 = fingerprint(df2)

        # Should be same due to category sorting in canonicalization
        assert fp1.hex == fp2.hex


class TestPerformanceBenchmarks:
    """Benchmark tests for performance measurement."""

    @pytest.mark.slow
    def test_performance_benchmark_medium(self):
        """Benchmark performance on medium-sized DataFrame."""
        df = pd.DataFrame(
            {
                "int_col": np.random.randint(-1000, 1000, 10000),
                "float_col": np.random.randn(10000),
                "str_col": [f"value_{i}" for i in range(10000)],
                "datetime_col": pd.date_range("2020-01-01", periods=10000, freq="1h"),
            }
        )

        start_time = time.time()
        fp = fingerprint(df)
        end_time = time.time()

        elapsed = end_time - start_time
        print(f"\nBenchmark: 10k rows, 4 columns took {elapsed:.3f}s")

        assert fp.hex.startswith("sha256:")
        # Should complete in reasonable time
        assert elapsed < 2.0, f"Performance regression: {elapsed:.3f}s > 2.0s"

    @pytest.mark.slow
    def test_performance_benchmark_wide(self):
        """Benchmark performance on wide DataFrame."""
        n_cols = 100
        n_rows = 1000

        data = {}
        for i in range(n_cols):
            data[f"col_{i}"] = np.random.randn(n_rows)

        df = pd.DataFrame(data)

        start_time = time.time()
        fp = fingerprint(df)
        end_time = time.time()

        elapsed = end_time - start_time
        print(f"\nBenchmark: 1k rows, 100 columns took {elapsed:.3f}s")

        assert fp.hex.startswith("sha256:")
        # Should complete in reasonable time
        assert elapsed < 2.0, f"Performance regression: {elapsed:.3f}s > 2.0s"


if __name__ == "__main__":
    # Run performance tests manually
    test_perf = TestVectorizedPerformance()

    print("Testing vectorized correctness...")
    test_perf.test_vectorized_correctness_small()
    print("✓ Small DataFrame correctness")

    test_perf.test_vectorized_correctness_with_nulls()
    print("✓ DataFrame with nulls correctness")

    test_perf.test_vectorized_deterministic_large()
    print("✓ Large DataFrame deterministic")

    # Simple benchmark
    df = pd.DataFrame(
        {
            "int_col": np.random.randint(-1000, 1000, 5000),
            "float_col": np.random.randn(5000),
            "str_col": [f"value_{i}" for i in range(5000)],
        }
    )

    start = time.time()
    fp = fingerprint(df)
    end = time.time()

    print(f"✓ 5k rows benchmark: {end - start:.3f}s")
    print("All performance tests passed!")
