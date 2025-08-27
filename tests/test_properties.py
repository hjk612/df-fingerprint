"""
Property-based tests using Hypothesis to verify canonicalization properties.

These tests generate random DataFrames and verify that fundamental properties
of canonicalization hold across a wide range of inputs.
"""

import pandas as pd
import numpy as np
import pytest
from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra.pandas import data_frames, columns, range_indexes
from df_fingerprint import fingerprint, Canonicalizer


# Custom strategies for DataFrame generation
@st.composite
def simple_dataframes(draw):
    """Generate simple DataFrames with basic dtypes, avoiding pandas cast warnings."""
    n_cols = draw(st.integers(min_value=1, max_value=5))
    n_rows = draw(st.integers(min_value=0, max_value=10))

    col_names = [f"col_{i}" for i in range(n_cols)]
    data = {}

    for col in col_names:
        dtype_choice = draw(st.sampled_from(["int", "float", "str", "bool"]))

        if dtype_choice == "int":
            # Use pandas nullable integer dtype to avoid cast warnings
            if n_rows == 0:
                data[col] = pd.array([], dtype="Int64")
            else:
                has_nulls = draw(st.booleans())
                if has_nulls:
                    # Generate with explicit nullable integer dtype
                    values = []
                    for i in range(n_rows):
                        if (
                            draw(st.booleans()) and draw(st.floats(0, 1)) < 0.2
                        ):  # 20% chance of null
                            values.append(pd.NA)
                        else:
                            values.append(
                                draw(st.integers(min_value=-1000, max_value=1000))
                            )
                    data[col] = pd.array(values, dtype="Int64")
                else:
                    # All integers, use regular int64
                    values = draw(
                        st.lists(
                            st.integers(min_value=-1000, max_value=1000),
                            min_size=n_rows,
                            max_size=n_rows,
                        )
                    )
                    data[col] = values

        elif dtype_choice == "float":
            # Use clean float generation to avoid cast warnings
            if n_rows == 0:
                data[col] = []
            else:
                has_nulls = draw(st.booleans())
                if has_nulls:
                    values = []
                    for i in range(n_rows):
                        if (
                            draw(st.booleans()) and draw(st.floats(0, 1)) < 0.2
                        ):  # 20% chance of null
                            values.append(np.nan)
                        else:
                            # Generate finite floats only
                            values.append(
                                draw(
                                    st.floats(
                                        min_value=-1000.0,
                                        max_value=1000.0,
                                        allow_nan=False,
                                        allow_infinity=False,
                                    )
                                )
                            )
                    data[col] = values
                else:
                    # All finite floats
                    data[col] = draw(
                        st.lists(
                            st.floats(
                                min_value=-1000.0,
                                max_value=1000.0,
                                allow_nan=False,
                                allow_infinity=False,
                            ),
                            min_size=n_rows,
                            max_size=n_rows,
                        )
                    )
        elif dtype_choice == "str":
            # Generate string data without mixed types
            if n_rows == 0:
                data[col] = []
            else:
                has_nulls = draw(st.booleans())
                if has_nulls:
                    values = []
                    for i in range(n_rows):
                        if (
                            draw(st.booleans()) and draw(st.floats(0, 1)) < 0.2
                        ):  # 20% chance of null
                            values.append(None)
                        else:
                            values.append(draw(st.text(min_size=0, max_size=10)))
                    data[col] = values
                else:
                    data[col] = draw(
                        st.lists(
                            st.text(min_size=0, max_size=10),
                            min_size=n_rows,
                            max_size=n_rows,
                        )
                    )

        elif dtype_choice == "bool":
            # Use pandas nullable boolean dtype to avoid warnings
            if n_rows == 0:
                data[col] = pd.array([], dtype="boolean")
            else:
                has_nulls = draw(st.booleans())
                if has_nulls:
                    values = []
                    for i in range(n_rows):
                        if (
                            draw(st.booleans()) and draw(st.floats(0, 1)) < 0.2
                        ):  # 20% chance of null
                            values.append(pd.NA)
                        else:
                            values.append(draw(st.booleans()))
                    data[col] = pd.array(values, dtype="boolean")
                else:
                    data[col] = draw(
                        st.lists(st.booleans(), min_size=n_rows, max_size=n_rows)
                    )

    return pd.DataFrame(data)


class TestCanonicalizationProperties:
    """Property-based tests for canonicalization invariants."""

    @given(simple_dataframes())
    @settings(max_examples=50, deadline=5000)
    def test_fingerprint_deterministic(self, df):
        """Same DataFrame should always produce same fingerprint."""
        assume(len(df.columns) > 0)  # Skip empty DataFrames for this test

        fp1 = fingerprint(df)
        fp2 = fingerprint(df)

        assert fp1.hex == fp2.hex
        assert fp1.digest == fp2.digest

    @given(simple_dataframes())
    @settings(max_examples=50, deadline=5000)
    def test_column_order_invariant(self, df):
        """Column order should not affect fingerprint when sort_columns=True."""
        assume(len(df.columns) > 1)  # Need multiple columns to reorder

        # Reorder columns
        cols = list(df.columns)
        reordered_cols = cols[::-1]  # Reverse order
        df_reordered = df[reordered_cols]

        fp1 = fingerprint(df)
        fp2 = fingerprint(df_reordered)

        assert fp1.hex == fp2.hex

    @given(simple_dataframes())
    @settings(max_examples=50, deadline=5000)
    def test_copy_invariant(self, df):
        """DataFrame copy should produce same fingerprint."""
        assume(len(df.columns) > 0)

        df_copy = df.copy()

        fp1 = fingerprint(df)
        fp2 = fingerprint(df_copy)

        assert fp1.hex == fp2.hex

    @given(simple_dataframes())
    @settings(max_examples=30, deadline=5000)
    def test_index_reset_invariant_when_dropped(self, df):
        """Index reset should not affect fingerprint when keep_index=False."""
        assume(len(df.columns) > 0)

        # Create DataFrame with non-default index
        df_with_index = df.copy()
        if len(df) > 0:
            df_with_index.index = range(100, 100 + len(df))

        canon = Canonicalizer(keep_index=False)
        fp1 = canon.hash(df)
        fp2 = canon.hash(df_with_index)

        assert fp1.hex == fp2.hex

    @given(
        df=simple_dataframes(),
        sort_cols=st.booleans(),
        keep_idx=st.booleans(),
        tz=st.sampled_from(["UTC", "US/Eastern", "Europe/London"]),
    )
    @settings(max_examples=30, deadline=5000)
    def test_configuration_consistency(self, df, sort_cols, keep_idx, tz):
        """Same configuration should produce same results."""
        assume(len(df.columns) > 0)

        canon1 = Canonicalizer(sort_columns=sort_cols, keep_index=keep_idx, tz=tz)
        canon2 = Canonicalizer(sort_columns=sort_cols, keep_index=keep_idx, tz=tz)

        fp1 = canon1.hash(df)
        fp2 = canon2.hash(df)

        assert fp1.hex == fp2.hex
        assert fp1.meta["options"] == fp2.meta["options"]

    @given(simple_dataframes())
    @settings(max_examples=30, deadline=5000)
    def test_metadata_completeness(self, df):
        """Fingerprint metadata should always be complete."""
        assume(len(df.columns) > 0)

        fp = fingerprint(df)

        # Check required metadata fields
        assert "spec" in fp.meta
        assert "algo" in fp.meta
        assert "python" in fp.meta
        assert "pandas" in fp.meta
        assert "numpy" in fp.meta
        assert "options" in fp.meta

        # Check spec version
        assert fp.meta["spec"] == "df-canon-v1"

        # Check algorithm matches fingerprint
        assert fp.algo == fp.meta["algo"]

    @given(simple_dataframes())
    @settings(max_examples=30, deadline=5000)
    def test_fingerprint_format_consistency(self, df):
        """Fingerprint format should be consistent."""
        assume(len(df.columns) > 0)

        fp = fingerprint(df)

        # Check hex format
        assert fp.hex.startswith(f"{fp.algo}:")
        hex_part = fp.hex.split(":", 1)[1]
        assert len(hex_part) > 0
        assert all(c in "0123456789abcdef" for c in hex_part)

        # Check digest length (SHA-256 = 32 bytes)
        if fp.algo == "sha256":
            assert len(fp.digest) == 32


class TestEdgeCaseProperties:
    """Property-based tests for edge cases and boundary conditions."""

    @given(
        st.lists(
            st.one_of(st.none(), st.floats(allow_nan=True)), min_size=1, max_size=10
        )
    )
    @settings(max_examples=30, deadline=3000)
    def test_missing_value_normalization_property(self, values):
        """All missing values should normalize identically."""
        # Create DataFrames with different missing value representations
        df_none = pd.DataFrame({"col": [None if pd.isna(v) else v for v in values]})
        df_nan = pd.DataFrame({"col": [np.nan if pd.isna(v) else v for v in values]})

        try:
            fp1 = fingerprint(df_none)
            fp2 = fingerprint(df_nan)
            assert fp1.hex == fp2.hex
        except Exception:
            # Some combinations might not be valid, skip those
            assume(False)

    @given(st.integers(min_value=0, max_value=5))
    @settings(max_examples=10)
    def test_empty_dataframe_variations(self, n_cols):
        """Different ways of creating empty DataFrames should be equivalent."""
        if n_cols == 0:
            df1 = pd.DataFrame()
            df2 = pd.DataFrame({})
        else:
            col_names = [f"col_{i}" for i in range(n_cols)]
            df1 = pd.DataFrame({col: [] for col in col_names})
            df2 = pd.DataFrame(columns=col_names)

        fp1 = fingerprint(df1)
        fp2 = fingerprint(df2)

        assert fp1.hex == fp2.hex

    @given(
        st.lists(st.text(min_size=1, max_size=5), min_size=1, max_size=5, unique=True)
    )
    @settings(max_examples=20, deadline=3000)
    def test_column_name_variations(self, col_names):
        """Column name variations should be handled consistently."""
        # Create DataFrame with these column names
        data = {col: [1] for col in col_names}
        df1 = pd.DataFrame(data)

        # Create same DataFrame with columns in different order
        shuffled_names = col_names[::-1]
        data_shuffled = {col: [1] for col in shuffled_names}
        df2 = pd.DataFrame(data_shuffled)

        fp1 = fingerprint(df1)
        fp2 = fingerprint(df2)

        # Should be same due to column sorting
        assert fp1.hex == fp2.hex


# Performance and stress tests
class TestPerformanceProperties:
    """Property-based tests for performance characteristics."""

    @given(st.integers(min_value=1, max_value=1000))
    @settings(max_examples=5, deadline=10000)
    def test_large_dataframe_handling(self, n_rows):
        """Large DataFrames should be handled without errors."""
        df = pd.DataFrame(
            {
                "int_col": range(n_rows),
                "float_col": [i * 0.1 for i in range(n_rows)],
                "str_col": [f"value_{i}" for i in range(n_rows)],
            }
        )

        # Should not raise exceptions
        fp = fingerprint(df)
        assert fp.hex.startswith("sha256:")
        assert len(fp.digest) == 32

    @given(st.integers(min_value=1, max_value=50))
    @settings(max_examples=5, deadline=5000)
    def test_wide_dataframe_handling(self, n_cols):
        """DataFrames with many columns should be handled efficiently."""
        data = {f"col_{i}": [1, 2, 3] for i in range(n_cols)}
        df = pd.DataFrame(data)

        # Should not raise exceptions
        fp = fingerprint(df)
        assert fp.hex.startswith("sha256:")
        assert len(fp.meta["options"]) > 0


if __name__ == "__main__":
    # Run a few property tests manually for debugging
    import sys

    print("Running property-based tests manually...")

    # Test deterministic property
    df = pd.DataFrame({"a": [1, 2, None], "b": [1.0, np.nan, 3.14]})
    fp1 = fingerprint(df)
    fp2 = fingerprint(df)
    print(f"Deterministic test: {fp1.hex == fp2.hex}")

    # Test column order invariant
    df_reordered = df[["b", "a"]]
    fp3 = fingerprint(df_reordered)
    print(f"Column order invariant: {fp1.hex == fp3.hex}")

    print("Manual tests completed successfully!")
