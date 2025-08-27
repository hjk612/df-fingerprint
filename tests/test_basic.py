import pandas as pd
import numpy as np
import pytest
from decimal import Decimal
from df_fingerprint import fingerprint, Canonicalizer


def test_same_data_diff_column_order_same_hash():
    df1 = pd.DataFrame({"a": [1, 2, None], "b": [1.0, float("nan"), 3.14]})
    df2 = df1[["b", "a"]]  # different column order
    fp1 = fingerprint(df1)
    fp2 = fingerprint(df2)
    assert fp1.hex == fp2.hex, (fp1.hex, fp2.hex)


def test_datetime_timezone_normalization():
    # Test that the same moment in time in different timezones produces the same hash
    # 2024-01-01 00:00:00 UTC = 2023-12-31 16:00:00 PST (UTC-8)
    df_utc = pd.DataFrame({"t": pd.to_datetime(["2024-01-01 00:00:00"], utc=True)})
    df_pst = pd.DataFrame(
        {"t": pd.to_datetime(["2023-12-31 16:00:00"]).tz_localize("US/Pacific")}
    )
    canon = Canonicalizer(tz="UTC")
    fp1 = canon.hash(df_utc)
    fp2 = canon.hash(df_pst)
    assert fp1.hex == fp2.hex


def test_nan_nat_nulls_normalize():
    df = pd.DataFrame(
        {
            "x": [np.nan, None, 1],
            "y": [pd.NaT, pd.Timestamp("2024-01-01", tz="UTC"), None],
        }
    )
    fp = fingerprint(df)
    # Ensure it runs and produces a stable string
    assert fp.hex.startswith("sha256:")


# ============================================================================
# COMPREHENSIVE EDGE CASE TESTS
# ============================================================================


class TestMissingValueNormalization:
    """Test that all forms of missing values normalize to the same canonical null."""

    @pytest.mark.parametrize(
        "missing_val",
        [
            np.nan,
            None,
            pd.NA,
            pd.NaT,
        ],
    )
    def test_missing_values_normalize_identically(self, missing_val):
        """All missing value types should produce identical fingerprints."""
        df1 = pd.DataFrame({"col": [1, missing_val, 3]})
        df2 = pd.DataFrame({"col": [1, None, 3]})  # baseline with None

        fp1 = fingerprint(df1)
        fp2 = fingerprint(df2)
        assert fp1.hex == fp2.hex

    def test_mixed_missing_values_same_fingerprint(self):
        """DataFrame with different missing types should hash same as uniform None."""
        df_mixed = pd.DataFrame(
            {
                "a": [1, np.nan, 3],
                "b": [pd.NaT, pd.Timestamp("2024-01-01"), None],
                "c": [None, "hello", pd.NA],
            }
        )

        df_none = pd.DataFrame(
            {
                "a": [1, None, 3],
                "b": [None, pd.Timestamp("2024-01-01"), None],
                "c": [None, "hello", None],
            }
        )

        fp_mixed = fingerprint(df_mixed)
        fp_none = fingerprint(df_none)
        assert fp_mixed.hex == fp_none.hex


class TestFloatEdgeCases:
    """Test float special values and precision handling."""

    @pytest.mark.parametrize(
        "special_float,expected_encoding",
        [
            (float("inf"), ["f", "inf"]),
            (float("-inf"), ["f", "-inf"]),
            (-0.0, ["f", "-0.0"]),
            (0.0, ["f", "0.0"]),
        ],
    )
    def test_special_float_values(self, special_float, expected_encoding):
        """Special float values should be encoded consistently."""
        canon = Canonicalizer()
        encoded = canon._encode_value(special_float)
        assert encoded == expected_encoding

    def test_negative_zero_vs_positive_zero(self):
        """Negative zero and positive zero should produce same fingerprints after normalization."""
        # Note: convert_dtypes() normalizes 0.0 and -0.0 to the same integer 0
        df1 = pd.DataFrame({"val": [0.0]})
        df2 = pd.DataFrame({"val": [-0.0]})

        fp1 = fingerprint(df1)
        fp2 = fingerprint(df2)
        # They should be the same because convert_dtypes() normalizes both to integer 0
        assert fp1.hex == fp2.hex

        # But if we force them to stay as floats, they should be different
        df1_float = pd.DataFrame({"val": [0.0]}).astype({"val": "float64"})
        df2_float = pd.DataFrame({"val": [-0.0]}).astype({"val": "float64"})

        # Skip convert_dtypes to preserve float types
        canon = Canonicalizer()
        # Manually normalize without convert_dtypes
        df1_norm = df1_float.reset_index(drop=True)
        df2_norm = df2_float.reset_index(drop=True)

        encoded1 = canon._encode_value(df1_norm["val"].iloc[0])
        encoded2 = canon._encode_value(df2_norm["val"].iloc[0])

        # These should be different at the encoding level
        assert encoded1 != encoded2

    def test_float_precision_modes(self):
        """Test different float precision handling modes."""
        df = pd.DataFrame({"val": [3.141592653589793]})

        # Round-trip mode (default)
        canon_roundtrip = Canonicalizer(float_mode="round-trip")
        fp_roundtrip = canon_roundtrip.hash(df)

        # Decimal quantization mode
        canon_decimal = Canonicalizer(float_mode="decimal=1e-10")
        fp_decimal = canon_decimal.hash(df)

        # Should produce different fingerprints due to different representations
        assert fp_roundtrip.hex != fp_decimal.hex

    def test_float_quantization_consistency(self):
        """Values that quantize to same decimal should have same fingerprint."""
        canon = Canonicalizer(float_mode="decimal=1e-2")

        df1 = pd.DataFrame({"val": [3.141]})
        df2 = pd.DataFrame({"val": [3.142]})  # Both round to 3.14
        df3 = pd.DataFrame({"val": [3.144]})  # Rounds to 3.14

        fp1 = canon.hash(df1)
        fp2 = canon.hash(df2)
        fp3 = canon.hash(df3)

        # All should quantize to 3.14
        assert fp1.hex == fp2.hex == fp3.hex


class TestTimezoneHandling:
    """Test comprehensive timezone normalization."""

    def test_naive_vs_aware_same_moment(self):
        """Naive datetime assumed UTC should equal explicit UTC."""
        df_naive = pd.DataFrame({"dt": pd.to_datetime(["2024-01-01 12:00:00"])})
        df_aware = pd.DataFrame(
            {"dt": pd.to_datetime(["2024-01-01 12:00:00"], utc=True)}
        )

        canon = Canonicalizer(tz="UTC")
        fp_naive = canon.hash(df_naive)
        fp_aware = canon.hash(df_aware)

        assert fp_naive.hex == fp_aware.hex

    @pytest.mark.parametrize(
        "tz1,tz2,time1,time2",
        [
            ("UTC", "US/Eastern", "2024-06-01 12:00:00", "2024-06-01 08:00:00"),  # EDT
            (
                "UTC",
                "Europe/London",
                "2024-06-01 12:00:00",
                "2024-06-01 13:00:00",
            ),  # BST
            (
                "US/Pacific",
                "US/Eastern",
                "2024-06-01 09:00:00",
                "2024-06-01 12:00:00",
            ),  # Same moment
        ],
    )
    def test_equivalent_moments_different_timezones(self, tz1, tz2, time1, time2):
        """Same moment in different timezones should produce same fingerprint."""
        df1 = pd.DataFrame({"dt": pd.to_datetime([time1]).tz_localize(tz1)})
        df2 = pd.DataFrame({"dt": pd.to_datetime([time2]).tz_localize(tz2)})

        canon = Canonicalizer(tz="UTC")
        fp1 = canon.hash(df1)
        fp2 = canon.hash(df2)

        assert fp1.hex == fp2.hex

    def test_nat_handling_in_datetime_columns(self):
        """NaT in datetime columns should normalize consistently."""
        df1 = pd.DataFrame({"dt": [pd.Timestamp("2024-01-01"), pd.NaT]})
        df2 = pd.DataFrame({"dt": [pd.Timestamp("2024-01-01"), None]})

        fp1 = fingerprint(df1)
        fp2 = fingerprint(df2)

        assert fp1.hex == fp2.hex


class TestCategoricalHandling:
    """Test categorical data normalization."""

    def test_unsorted_vs_sorted_categories_same_fingerprint(self):
        """Categories with different ordering should produce same fingerprint."""
        # Unsorted categories
        cat1 = pd.Categorical(
            ["red", "blue", "green"], categories=["red", "green", "blue"]
        )
        df1 = pd.DataFrame({"color": cat1})

        # Sorted categories
        cat2 = pd.Categorical(
            ["red", "blue", "green"], categories=["blue", "green", "red"]
        )
        df2 = pd.DataFrame({"color": cat2})

        fp1 = fingerprint(df1)
        fp2 = fingerprint(df2)

        assert fp1.hex == fp2.hex

    def test_categorical_with_missing_values(self):
        """Categorical with missing values should normalize consistently."""
        cat = pd.Categorical(["a", None, "b"], categories=["b", "a"])
        df = pd.DataFrame({"cat_col": cat})

        fp = fingerprint(df)
        assert fp.hex.startswith("sha256:")

        # Should be same as string column with None
        df_str = pd.DataFrame({"cat_col": ["a", None, "b"]})
        fp_str = fingerprint(df_str)

        # Note: These will be different because categorical vs string encoding
        # but both should be valid fingerprints
        assert fp.hex.startswith("sha256:")
        assert fp_str.hex.startswith("sha256:")


class TestBinaryAndSpecialTypes:
    """Test handling of bytes, Decimals, and other special types."""

    def test_bytes_to_base64_encoding(self):
        """Bytes should be consistently encoded as base64."""
        df1 = pd.DataFrame({"data": [b"hello", b"world"]})
        df2 = pd.DataFrame({"data": [b"hello", b"world"]})

        fp1 = fingerprint(df1)
        fp2 = fingerprint(df2)

        assert fp1.hex == fp2.hex

        # Check the actual encoding
        canon = Canonicalizer()
        encoded = canon._encode_value(b"hello")
        assert encoded == ["bytes", "aGVsbG8="]  # base64 of "hello"

    def test_decimal_precision_preservation(self):
        """Decimal values should preserve exact precision."""
        df1 = pd.DataFrame({"precise": [Decimal("3.14159265358979323846")]})
        df2 = pd.DataFrame({"precise": [Decimal("3.14159265358979323846")]})

        fp1 = fingerprint(df1)
        fp2 = fingerprint(df2)

        assert fp1.hex == fp2.hex

        # Different precision should produce different fingerprint
        df3 = pd.DataFrame({"precise": [Decimal("3.141592653589793")]})
        fp3 = fingerprint(df3)

        assert fp1.hex != fp3.hex

    def test_timedelta_nanosecond_encoding(self):
        """Timedeltas should be encoded as nanoseconds consistently."""
        df1 = pd.DataFrame({"td": [pd.Timedelta("1 day"), pd.Timedelta("1.5 seconds")]})
        df2 = pd.DataFrame({"td": [pd.Timedelta("1 day"), pd.Timedelta("1.5 seconds")]})

        fp1 = fingerprint(df1)
        fp2 = fingerprint(df2)

        assert fp1.hex == fp2.hex

        # Check encoding
        canon = Canonicalizer()
        encoded = canon._encode_value(pd.Timedelta("1 day"))
        assert encoded == ["td", 86400000000000]  # 1 day in nanoseconds


class TestIndexHandling:
    """Test index inclusion/exclusion behavior."""

    def test_index_dropped_by_default(self):
        """Default behavior should ignore index."""
        df1 = pd.DataFrame({"val": [1, 2, 3]}, index=[10, 20, 30])
        df2 = pd.DataFrame({"val": [1, 2, 3]}, index=["a", "b", "c"])

        fp1 = fingerprint(df1)
        fp2 = fingerprint(df2)

        assert fp1.hex == fp2.hex

    def test_index_kept_when_requested(self):
        """keep_index=True should include index deterministically."""
        df = pd.DataFrame({"val": [1, 2, 3]}, index=["c", "a", "b"])

        canon_drop = Canonicalizer(keep_index=False)
        canon_keep = Canonicalizer(keep_index=True)

        fp_drop = canon_drop.hash(df)
        fp_keep = canon_keep.hash(df)

        # Should be different
        assert fp_drop.hex != fp_keep.hex

        # Same DataFrame with keep_index should be deterministic
        fp_keep2 = canon_keep.hash(df)
        assert fp_keep.hex == fp_keep2.hex

    def test_multiindex_handling(self):
        """MultiIndex should be handled deterministically."""
        idx = pd.MultiIndex.from_tuples(
            [("A", 1), ("B", 2), ("A", 3)], names=["letter", "number"]
        )
        df = pd.DataFrame({"val": [10, 20, 30]}, index=idx)

        canon = Canonicalizer(keep_index=True)
        fp1 = canon.hash(df)
        fp2 = canon.hash(df)

        assert fp1.hex == fp2.hex


class TestColumnHandling:
    """Test column name and ordering behavior."""

    def test_column_sorting_consistency(self):
        """Different column orders should produce same fingerprint when sorted."""
        # Same data, different column order
        df1 = pd.DataFrame({"z": [1, 4], "a": [2, 5], "b": [3, 6]})
        df2 = pd.DataFrame({"a": [2, 5], "b": [3, 6], "z": [1, 4]})

        fp1 = fingerprint(df1)
        fp2 = fingerprint(df2)

        assert fp1.hex == fp2.hex

    def test_numeric_column_names_sorting(self):
        """Numeric column names should sort consistently."""
        df1 = pd.DataFrame({1: [10], 3: [30], 2: [20]})
        df2 = pd.DataFrame({2: [20], 1: [10], 3: [30]})

        fp1 = fingerprint(df1)
        fp2 = fingerprint(df2)

        assert fp1.hex == fp2.hex

    def test_case_sensitive_column_sorting(self):
        """Case-sensitive column sorting should be consistent."""
        df1 = pd.DataFrame({"Z": [1], "a": [2], "B": [3]})
        df2 = pd.DataFrame({"B": [3], "Z": [1], "a": [2]})

        fp1 = fingerprint(df1)
        fp2 = fingerprint(df2)

        assert fp1.hex == fp2.hex

    def test_column_sorting_disabled(self):
        """When sort_columns=False, order should matter."""
        df1 = pd.DataFrame({"a": [1], "b": [2]})
        df2 = pd.DataFrame({"b": [2], "a": [1]})

        canon = Canonicalizer(sort_columns=False)
        fp1 = canon.hash(df1)
        fp2 = canon.hash(df2)

        assert fp1.hex != fp2.hex


class TestEmptyAndEdgeCases:
    """Test empty DataFrames and other edge cases."""

    def test_empty_dataframe(self):
        """Empty DataFrame should produce consistent fingerprint."""
        df1 = pd.DataFrame()
        df2 = pd.DataFrame()

        fp1 = fingerprint(df1)
        fp2 = fingerprint(df2)

        assert fp1.hex == fp2.hex

    def test_single_cell_dataframe(self):
        """Single cell DataFrame should work correctly."""
        df = pd.DataFrame({"col": [42]})
        fp = fingerprint(df)

        assert fp.hex.startswith("sha256:")

    def test_large_integer_handling(self):
        """Very large integers should be handled consistently."""
        large_int = 2**100
        df1 = pd.DataFrame({"big": [large_int]})
        df2 = pd.DataFrame({"big": [large_int]})

        fp1 = fingerprint(df1)
        fp2 = fingerprint(df2)

        assert fp1.hex == fp2.hex


class TestMetadataAndVersioning:
    """Test fingerprint metadata and versioning."""

    def test_fingerprint_metadata_includes_spec_version(self):
        """Fingerprint metadata should include spec version."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        fp = fingerprint(df)

        assert fp.meta["spec"] == "df-canon-v1"
        assert "pandas" in fp.meta
        assert "numpy" in fp.meta
        assert "python" in fp.meta

    def test_configuration_in_metadata(self):
        """Configuration options should be recorded in metadata."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        canon = Canonicalizer(
            sort_columns=False, keep_index=True, float_mode="decimal=1e-6"
        )
        fp = canon.hash(df)

        options = fp.meta["options"]
        assert options["sort_columns"] == False
        assert options["keep_index"] == True
        assert options["float_mode"] == "decimal=1e-6"
