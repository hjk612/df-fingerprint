# df-fingerprint

A tiny library that turns any pandas `DataFrame` into a **canonical byte representation** and returns a **stable fingerprint** you can trust for caching, lineage, and reproducibility across machines and pandas versions.

## Why?
`pd.util.hash_pandas_object` and generic object hashing aren't designed as a **cross-environment standard**. The same logical data can hash differently due to column order, dtype quirks, NaN vs NaT, float formatting, or timezone handling.

**df-fingerprint** defines a small **canonicalization spec (df-canon v1)** and implements it, then hashes the result (SHA-256 by default). If the data means the same, the fingerprint is the same.

## Install
```bash
pip install df-fingerprint
```

## Motivating Examples

### Cache Keys for Feature Engineering
```python
import pandas as pd
from df_fingerprint import fingerprint

# Feature engineering pipeline
def create_features(raw_data):
    # Complex transformations...
    return processed_df

# Use fingerprint as cache key
raw_df = pd.read_csv("data.csv")
cache_key = fingerprint(raw_df).hex

if cache_key in feature_cache:
    features = feature_cache[cache_key]
else:
    features = create_features(raw_df)
    feature_cache[cache_key] = features
```

### Verifying Production vs Local Datasets
```python
# Local development
local_df = pd.read_parquet("local_data.parquet")
local_fp = fingerprint(local_df)

# Production environment
prod_df = pd.read_parquet("s3://bucket/prod_data.parquet")
prod_fp = fingerprint(prod_df)

# Verify data integrity across environments
assert local_fp.hex == prod_fp.hex, "Data mismatch between local and prod!"
```

### Experiment Lineage and Reproducibility
```python
# Track data lineage in ML experiments
experiment_metadata = {
    "model_version": "v2.1",
    "training_data": fingerprint(train_df).hex,
    "validation_data": fingerprint(val_df).hex,
    "test_data": fingerprint(test_df).hex,
    "timestamp": datetime.now().isoformat()
}

# Later: verify exact same data was used
def verify_experiment_data(train_df, val_df, test_df, metadata):
    assert fingerprint(train_df).hex == metadata["training_data"]
    assert fingerprint(val_df).hex == metadata["validation_data"] 
    assert fingerprint(test_df).hex == metadata["test_data"]
    print("✓ Experiment data verified - exact reproduction possible")
```

### Dataset Versioning and Change Detection
```python
# Monitor data drift and changes
def monitor_dataset_changes(new_df, baseline_fingerprint):
    current_fp = fingerprint(new_df)
    
    if current_fp.hex != baseline_fingerprint:
        print(f"⚠️  Dataset changed!")
        print(f"Previous: {baseline_fingerprint}")
        print(f"Current:  {current_fp.hex}")
        return False
    else:
        print("✓ Dataset unchanged")
        return True

# Usage in data pipeline
baseline = "sha256:abc123..."
is_same = monitor_dataset_changes(daily_data, baseline)
```

## Quick Start
```python
import pandas as pd
from df_fingerprint import fingerprint, Canonicalizer

df = pd.DataFrame({
    "a": [1, 2, None],
    "b": [1.0, float("nan"), 3.14],
})

fp = fingerprint(df)
print(fp.hex)      # e.g., "sha256:0e7c7f..."
print(fp.meta)     # spec + environment info
```

## API Reference

### `fingerprint(df, *, algo="sha256", canonicalizer=None)`
Generate a fingerprint for a DataFrame using default canonicalization.

**Parameters:**
- `df` (pd.DataFrame): The DataFrame to fingerprint
- `algo` (str): Hash algorithm - "sha256" (default) or "blake3"
- `canonicalizer` (Canonicalizer, optional): Custom canonicalizer instance

**Returns:**
- `Fingerprint`: Object with `.hex`, `.digest`, `.algo`, and `.meta` attributes

```python
# Basic usage
fp = fingerprint(df)
print(fp.hex)        # "sha256:abc123..."
print(fp.digest)     # Raw bytes
print(fp.algo)       # "sha256"
print(fp.meta)       # Metadata dict

# Custom algorithm
fp_blake = fingerprint(df, algo="blake3")
```

### `Canonicalizer(sort_columns=True, keep_index=False, tz="UTC", float_mode="round-trip", json_mode="deterministic")`
Configurable canonicalization engine.

**Parameters:**
- `sort_columns` (bool): Sort columns alphabetically (default: True)
- `keep_index` (bool): Include index in fingerprint (default: False)  
- `tz` (str): Target timezone for datetime normalization (default: "UTC")
- `float_mode` (str): Float precision handling - "round-trip" or "decimal=1e-12"
- `json_mode` (str): JSON serialization mode (default: "deterministic")

**Methods:**
- `.hash(df, algo="sha256")`: Generate fingerprint
- `.canonicalize(df)`: Get canonical representation as dict

```python
# Custom canonicalizer
canon = Canonicalizer(
    sort_columns=False,    # Preserve column order
    keep_index=True,       # Include index
    tz="US/Eastern",       # Eastern timezone
    float_mode="decimal=1e-10"  # Custom precision
)

fp = canon.hash(df)
# or
fp = fingerprint(df, canonicalizer=canon)
```

### Configuration Options

#### Column Handling
```python
# Preserve original column order
canon = Canonicalizer(sort_columns=False)

# Include index in fingerprint
canon = Canonicalizer(keep_index=True)
```

#### Timezone Normalization
```python
# Convert all datetimes to specific timezone
canon = Canonicalizer(tz="Europe/London")
canon = Canonicalizer(tz="US/Pacific")
```

#### Float Precision
```python
# High-precision round-trip (default)
canon = Canonicalizer(float_mode="round-trip")

# Fixed decimal precision
canon = Canonicalizer(float_mode="decimal=1e-12")
canon = Canonicalizer(float_mode="decimal=1e-6")
```

## Performance
**Vectorized column-wise encoding** for excellent performance on large DataFrames:
- **150k+ rows/second** on typical mixed-type DataFrames
- **1M+ rows/second** on homogeneous numeric data
- **Memory efficient** - processes columns independently
- **Scales linearly** with DataFrame size

### Benchmarks
```python
# Performance test on your data
import time
from df_fingerprint import fingerprint

start = time.time()
fp = fingerprint(large_df)  # Your DataFrame
elapsed = time.time() - start

rows_per_sec = len(large_df) / elapsed
print(f"Processed {rows_per_sec:,.0f} rows/second")
```

## Design Notes

### Canonicalization Specification (df-canon v1)

**df-fingerprint** implements the `df-canon v1` specification for deterministic DataFrame canonicalization. The spec ensures that logically equivalent DataFrames produce identical fingerprints regardless of:

- Column order (when `sort_columns=True`)
- Index values (when `keep_index=False`) 
- Dtype variations (int32 vs int64, object vs string)
- Missing value representations (NaN, None, NaT, pd.NA)
- Timezone representations (naive vs aware, different zones for same moment)
- Categorical category ordering
- Float precision artifacts

### Canonicalization Rules

1. **Index Handling**: Drop index by default (or include deterministically)
2. **Column Sorting**: Sort columns by name (case-sensitive) unless disabled
3. **Dtype Normalization**: Convert to standardized nullable types
4. **DateTime Normalization**: UTC timezone, ISO-8601 with microseconds, NaT → null
5. **Missing Value Normalization**: All missing values → JSON null
6. **Categorical Normalization**: Sorted categories with stable codes
7. **Binary Data**: Bytes → base64 encoding
8. **Decimal Precision**: Decimal → string with full precision
9. **Type Tagging**: bool/int/float → type-tagged JSON values
10. **Serialization**: Deterministic JSON (sorted keys, strict separators)

For complete specification details, see [`df_fingerprint/spec.py`](df_fingerprint/spec.py).

### Hash Algorithm Support

- **SHA-256** (default): Cryptographically secure, widely supported
- **BLAKE3**: High-performance alternative (requires `blake3` package)

```python
# SHA-256 (default)
fp = fingerprint(df, algo="sha256")

# BLAKE3 (faster, requires: pip install blake3)
fp = fingerprint(df, algo="blake3")
```

## Compatibility Guarantees

### df-canon v1 Specification
**df-fingerprint** implements the `df-canon v1` canonicalization specification, which provides strong compatibility guarantees:

#### ✅ **Stable Across Environments**
- Same fingerprint on different machines (Linux, macOS, Windows)
- Same fingerprint across Python versions (3.9+)
- Same fingerprint across pandas versions (1.5+)
- Same fingerprint regardless of numpy/pandas installation method

#### ✅ **Stable Across Time**
- Fingerprints remain valid indefinitely
- No breaking changes within df-canon v1
- Specification is frozen and versioned

#### ✅ **Deterministic Canonicalization**
- Column order independence (when `sort_columns=True`)
- Index independence (when `keep_index=False`)
- Dtype normalization (int32 → Int64, object → string)
- Missing value normalization (NaN/None/NaT/pd.NA → null)
- Timezone normalization (all datetimes → UTC)

#### 🔄 **Version Migration**
When df-canon v2 is released (future):
- v1 fingerprints remain valid
- Migration tools will be provided
- Backward compatibility maintained

### Specification Compliance
```python
# Check specification version
fp = fingerprint(df)
assert fp.meta["spec"] == "df-canon-v1"

# Verify environment info
print(fp.meta["python"])    # Python version
print(fp.meta["pandas"])    # Pandas version  
print(fp.meta["numpy"])     # NumPy version
print(fp.meta["options"])   # Canonicalizer settings
```

### Cross-Language Compatibility
The df-canon v1 specification is designed for cross-language implementation:
- **JSON-based canonical representation** 
- **Deterministic serialization rules**
- **Standardized type encoding**
- **Precise floating-point handling**

Future implementations in R, Julia, and other languages will produce identical fingerprints for equivalent data.

## Advanced Usage

### Custom Canonicalization
```python
# Preserve column order and include index
canon = Canonicalizer(
    sort_columns=False,
    keep_index=True,
    tz="US/Eastern"
)

# Get canonical representation without hashing
canonical_dict = canon.canonicalize(df)
print(canonical_dict["spec"])     # "df-canon-v1"
print(canonical_dict["columns"])  # Column names
print(canonical_dict["rows"])     # Encoded row data

# Hash with custom algorithm
fp = canon.hash(df, algo="blake3")
```

### Integration Examples

#### MLflow Integration
```python
import mlflow
from df_fingerprint import fingerprint

# Log dataset fingerprints with experiments
with mlflow.start_run():
    train_fp = fingerprint(train_df).hex
    val_fp = fingerprint(val_df).hex
    
    mlflow.log_param("train_data_fingerprint", train_fp)
    mlflow.log_param("val_data_fingerprint", val_fp)
    
    # Train model...
    mlflow.log_model(model, "model")
```

#### DVC Integration
```python
# Add fingerprints to DVC pipeline metadata
import yaml
from df_fingerprint import fingerprint

# Generate fingerprints
data_fp = fingerprint(processed_data).hex

# Update dvc.yaml
pipeline_config = {
    "stages": {
        "process_data": {
            "cmd": "python process.py",
            "deps": ["raw_data.csv"],
            "outs": ["processed_data.parquet"],
            "meta": {
                "data_fingerprint": data_fp,
                "spec_version": "df-canon-v1"
            }
        }
    }
}
```

#### Prefect Integration
```python
from prefect import task, flow
from df_fingerprint import fingerprint

@task
def validate_data_consistency(df, expected_fingerprint):
    current_fp = fingerprint(df).hex
    if current_fp != expected_fingerprint:
        raise ValueError(f"Data changed! Expected {expected_fingerprint}, got {current_fp}")
    return df

@flow
def data_pipeline():
    raw_data = extract_data()
    validated_data = validate_data_consistency(raw_data, "sha256:abc123...")
    processed_data = transform_data(validated_data)
    load_data(processed_data)
```

## Troubleshooting

### Common Issues

#### Different fingerprints for "same" data
```python
# Check canonicalization settings
fp1 = fingerprint(df1)
fp2 = fingerprint(df2)

if fp1.hex != fp2.hex:
    print("Settings:", fp1.meta["options"])
    print("Spec version:", fp1.meta["spec"])
    
    # Compare canonical representations
    canon = Canonicalizer()
    dict1 = canon.canonicalize(df1)
    dict2 = canon.canonicalize(df2)
    
    # Check differences
    print("Columns match:", dict1["columns"] == dict2["columns"])
    print("Row count:", len(dict1["rows"]), "vs", len(dict2["rows"]))
```

#### Performance optimization
```python
# For repeated fingerprinting with same settings
canon = Canonicalizer()  # Reuse instance

fingerprints = []
for df in dataframes:
    fp = canon.hash(df)  # Faster than fingerprint(df)
    fingerprints.append(fp.hex)
```

## Roadmap
- **CBOR canonical encoding** (RFC 8949 variations)
- **Additional hash algorithms** (SHA-3, BLAKE2)
- **Polars DataFrame support**
- **Cross-language implementations** (R, Julia)
- **Streaming fingerprints** for large datasets
- **Schema fingerprints** (structure without data)

## Contributing
Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License
MIT
