from __future__ import annotations
import json
import math
import base64
import hashlib
import platform
import warnings
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_timedelta64_dtype,
)
from decimal import Decimal, getcontext, ROUND_HALF_EVEN

from ._version import SPEC_VERSION


@dataclass(frozen=True)
class Fingerprint:
    algo: str
    hex: str
    digest: bytes
    meta: Dict[str, Any]

    def __str__(self) -> str:
        return f"{self.algo}:{self.hex}"


class Canonicalizer:
    def __init__(
        self,
        sort_columns: bool = True,
        keep_index: bool = False,
        tz: str = "UTC",
        float_mode: str = "round-trip",  # or 'decimal=1e-12'
        json_mode: str = "deterministic",
    ) -> None:
        self.sort_columns = sort_columns
        self.keep_index = keep_index
        self.tz = tz
        self.float_mode = float_mode
        self.json_mode = json_mode

        # parse decimal precision if set
        self.decimal_quant = None
        if self.float_mode.startswith("decimal="):
            try:
                eps = self.float_mode.split("=", 1)[1]
                if eps.endswith("e-"):
                    # not expected format; ignore
                    pass
                self.decimal_quant = Decimal(eps)
            except Exception:
                self.decimal_quant = Decimal("1e-12")

    def _normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # Drop or materialize index
        if self.keep_index:
            idx_df = df.index.to_frame(index=False)
            idx_df.columns = [f"__index_level_{i}__" for i in range(idx_df.shape[1])]
            df = pd.concat([idx_df, df.reset_index(drop=True)], axis=1)
        else:
            df = df.reset_index(drop=True)

        # Column sorting
        if self.sort_columns:
            df = df.reindex(sorted(df.columns.astype(str)), axis=1)

        # Convert dtypes with error handling for large integers
        # Suppress pandas cast warnings during dtype conversion
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in cast",
                category=RuntimeWarning,
            )
            try:
                df = df.convert_dtypes()  # nullable types where possible
            except (OverflowError, TypeError):
                # Handle columns individually if convert_dtypes fails
                for col in df.columns:
                    try:
                        df[col] = df[col].convert_dtypes()
                    except (OverflowError, TypeError):
                        # Keep as object dtype for large integers or other problematic types
                        pass

        # Normalize datetimes to UTC
        for c in df.columns:
            s = df[c]
            if is_datetime64_any_dtype(s):
                # make timezone-aware UTC
                s = pd.to_datetime(s, utc=True, errors="coerce")
                df[c] = (
                    s.dt.tz_convert(self.tz)
                    if s.dt.tz is not None
                    else s.dt.tz_localize(self.tz)
                )

        return df

    def _encode_value(self, v: Any) -> List[Any]:
        # Missing
        if v is None or (isinstance(v, float) and math.isnan(v)) or pd.isna(v):
            return ["null", None]

        # numpy types -> python
        if isinstance(v, (np.integer,)):
            return ["i", int(v)]
        if isinstance(v, (np.floating,)):
            return self._encode_float(float(v))
        if isinstance(v, (np.bool_,)):
            return ["b", bool(v)]

        # pandas Timestamp
        if isinstance(v, pd.Timestamp):
            if pd.isna(v):
                return ["null", None]
            if v.tz is None:
                v = v.tz_localize(self.tz)
            else:
                v = v.tz_convert(self.tz)
            iso = v.isoformat(timespec="microseconds").replace("+00:00", "Z")
            return ["dt", iso]

        # Timedelta
        if isinstance(v, pd.Timedelta):
            if pd.isna(v):
                return ["null", None]
            # represent as total nanoseconds (deterministic)
            return ["td", int(v.value)]

        # Decimal
        if isinstance(v, Decimal):
            return ["dec", format(v, "f")]

        # bytes
        if isinstance(v, (bytes, bytearray, memoryview)):
            b64 = base64.b64encode(bytes(v)).decode("ascii")
            return ["bytes", b64]

        # bool/int/float
        if isinstance(v, bool):
            return ["b", v]
        if isinstance(v, int):
            return ["i", v]
        if isinstance(v, float):
            return self._encode_float(v)

        # categories are passed as their string representation by caller
        # strings and other objects
        return ["s", str(v)]

    def _encode_float(self, f: float) -> List[Any]:
        if math.isnan(f):
            return ["null", None]
        if self.float_mode == "round-trip":
            # Use Python repr for deterministic string, tagged as float
            s = repr(float(f))
            return ["f", s]
        elif self.float_mode.startswith("decimal="):
            getcontext().rounding = ROUND_HALF_EVEN
            q = self.decimal_quant or Decimal("1e-12")
            d = Decimal(str(f)).quantize(q)
            return ["f", format(d, "f")]
        else:
            # default fallback
            return ["f", repr(float(f))]

    def _encode_series_vectorized(self, s: pd.Series) -> List[List[Any]]:
        """Vectorized encoding of an entire Series for better performance."""
        if len(s) == 0:
            return []

        if isinstance(s.dtype, pd.CategoricalDtype):
            return self._encode_categorical_series(s)
        elif is_datetime64_any_dtype(s.dtype):
            return self._encode_datetime_series(s)
        elif is_timedelta64_dtype(s.dtype):
            return self._encode_timedelta_series(s)
        elif pd.api.types.is_integer_dtype(s.dtype):
            return self._encode_integer_series(s)
        elif pd.api.types.is_float_dtype(s.dtype):
            return self._encode_float_series(s)
        elif pd.api.types.is_bool_dtype(s.dtype):
            return self._encode_boolean_series(s)
        elif pd.api.types.is_string_dtype(s.dtype) or pd.api.types.is_object_dtype(
            s.dtype
        ):
            return self._encode_object_series(s)
        else:
            # Fallback to row-by-row for unknown types
            return [self._encode_value(v) for v in s]

    def _encode_categorical_series(self, s: pd.Series) -> List[List[Any]]:
        """Vectorized encoding for categorical series."""
        s = s.astype("category")
        s = s.cat.set_categories(sorted(map(str, s.cat.categories)))

        # Vectorized approach: use pandas operations
        result: List[List[Any]] = []
        for v in s:
            if pd.isna(v):
                result.append(["null", None])
            else:
                result.append(["cat", str(v)])
        return result

    def _encode_datetime_series(self, s: pd.Series) -> List[List[Any]]:
        """Vectorized encoding for datetime series."""
        # Convert to UTC if not already timezone-aware
        if s.dt.tz is None:
            s = s.dt.tz_localize(self.tz)
        else:
            s = s.dt.tz_convert(self.tz)

        # Build result using individual timestamp processing for precision
        result: List[List[Any]] = []
        for v in s:
            if pd.isna(v):
                result.append(["null", None])
            else:
                # Use the same logic as _encode_value for consistency
                iso = v.isoformat(timespec="microseconds").replace("+00:00", "Z")
                result.append(["dt", iso])
        return result

    def _encode_timedelta_series(self, s: pd.Series) -> List[List[Any]]:
        """Vectorized encoding for timedelta series."""
        result: List[List[Any]] = []
        for v in s:
            if pd.isna(v):
                result.append(["null", None])
            else:
                result.append(["td", int(v.value)])  # nanoseconds
        return result

    def _encode_integer_series(self, s: pd.Series) -> List[List[Any]]:
        """Vectorized encoding for integer series."""
        # Use pandas vectorized operations
        mask_na = pd.isna(s)

        result: List[List[Any]] = []
        for i, (is_na, val) in enumerate(zip(mask_na, s)):
            if is_na:
                result.append(["null", None])
            else:
                result.append(["i", int(val)])
        return result

    def _encode_float_series(self, s: pd.Series) -> List[List[Any]]:
        """Vectorized encoding for float series."""
        result: List[List[Any]] = []
        for v in s:
            if pd.isna(v):
                result.append(["null", None])
            else:
                result.append(self._encode_float(float(v)))
        return result

    def _encode_boolean_series(self, s: pd.Series) -> List[List[Any]]:
        """Vectorized encoding for boolean series."""
        # Use pandas vectorized operations
        mask_na = pd.isna(s)

        result: List[List[Any]] = []
        for i, (is_na, val) in enumerate(zip(mask_na, s)):
            if is_na:
                result.append(["null", None])
            else:
                result.append(["b", bool(val)])
        return result

    def _encode_object_series(self, s: pd.Series) -> List[List[Any]]:
        """Vectorized encoding for object/string series."""
        result: List[List[Any]] = []
        for v in s:
            if pd.isna(v):
                result.append(["null", None])
            elif isinstance(v, (bytes, bytearray, memoryview)):
                b64 = base64.b64encode(bytes(v)).decode("ascii")
                result.append(["bytes", b64])
            elif isinstance(v, Decimal):
                result.append(["dec", format(v, "f")])
            elif isinstance(v, bool):
                result.append(["b", v])
            elif isinstance(v, (int, np.integer)):
                result.append(["i", int(v)])
            elif isinstance(v, (float, np.floating)):
                result.append(self._encode_float(float(v)))
            else:
                result.append(["s", str(v)])
        return result

    def _encode_series(self, s: pd.Series) -> List[List[Any]]:
        """Legacy method for backward compatibility - delegates to vectorized version."""
        return self._encode_series_vectorized(s)

    def canonicalize(self, df: pd.DataFrame) -> Dict[str, Any]:
        df = self._normalize_df(df)

        # Ensure columns names are strings (determinism)
        columns = list(map(str, df.columns))

        # Vectorized encoding: process each column entirely, then transpose
        encoded_columns = {}
        for col in columns:
            encoded_columns[col] = self._encode_series_vectorized(df[col])

        # Transpose column-wise encoding to row-wise for JSON structure
        n_rows = len(df)
        data_rows: List[List[List[Any]]] = []
        for row_idx in range(n_rows):
            encoded_row = []
            for col in columns:
                encoded_row.append(encoded_columns[col][row_idx])
            data_rows.append(encoded_row)

        # Build canonical document
        doc = {
            "spec": SPEC_VERSION,
            "columns": columns,
            "rows": data_rows,
        }
        return doc

    def hash(self, df: pd.DataFrame, algo: str = "sha256") -> Fingerprint:
        doc = self.canonicalize(df)
        # Deterministic JSON serialization
        payload = json.dumps(
            doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")

        if algo.lower() == "sha256":
            h = hashlib.sha256(payload).digest()
            hexstr = hashlib.sha256(payload).hexdigest()
        elif algo.lower() == "blake3":
            try:
                import blake3  # type: ignore

                h = blake3.blake3(payload).digest()
                hexstr = blake3.blake3(payload).hexdigest()
                algo = "blake3"
            except Exception:
                # fallback to sha256 if blake3 unavailable
                h = hashlib.sha256(payload).digest()
                hexstr = hashlib.sha256(payload).hexdigest()
                algo = "sha256"
        else:
            # default to sha256
            h = hashlib.sha256(payload).digest()
            hexstr = hashlib.sha256(payload).hexdigest()
            algo = "sha256"

        meta = {
            "spec": SPEC_VERSION,
            "algo": algo,
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "options": {
                "sort_columns": self.sort_columns,
                "keep_index": self.keep_index,
                "tz": self.tz,
                "float_mode": self.float_mode,
                "json_mode": self.json_mode,
            },
        }

        return Fingerprint(algo=algo, hex=f"{algo}:{hexstr}", digest=h, meta=meta)


def fingerprint(
    df: pd.DataFrame,
    *,
    algo: str = "sha256",
    canonicalizer: Optional[Canonicalizer] = None,
) -> Fingerprint:
    canon = canonicalizer or Canonicalizer()
    return canon.hash(df, algo=algo)
