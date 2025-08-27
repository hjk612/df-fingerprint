"""
DataFrame Canonicalization Specification (df-canon v1)
=====================================================

RFC-style specification for stable, cross-environment DataFrame fingerprinting.

Abstract
--------
This specification defines a canonical representation format for pandas DataFrames
that produces identical fingerprints for logically equivalent data across different
environments, pandas versions, and machine architectures. The canonicalization
process normalizes structural, temporal, and type-related variations while
preserving semantic meaning.

1. Introduction
---------------

### 1.1 Problem Statement
Existing DataFrame hashing methods (e.g., pd.util.hash_pandas_object) are not
designed for cross-environment stability. The same logical data can produce
different hashes due to:
- Column ordering differences
- Dtype representation variations (int32 vs int64, object vs string)
- Timezone handling inconsistencies
- NaN/NaT representation differences
- Float precision and formatting variations
- Index inclusion/exclusion ambiguity

### 1.2 Design Goals
- **Deterministic**: Same logical data always produces same fingerprint
- **Cross-platform**: Identical results across OS, Python, pandas versions
- **Semantic preservation**: Logically equivalent data hashes identically
- **Configurable**: Allow controlled variations for different use cases
- **Extensible**: Support future enhancements via versioning

### 1.3 Versioning and Compatibility
This specification is versioned as "df-canon v1". Any changes that would alter
fingerprints for existing data MUST increment the major version (v2, v3, etc.).
Implementations MUST maintain backward compatibility within the same major version.

2. Canonicalization Process
---------------------------

### 2.1 Overview
The canonicalization process transforms a pandas DataFrame through these stages:
1. Index normalization
2. Column ordering
3. Dtype normalization
4. Value encoding with type tags
5. JSON serialization
6. Cryptographic hashing

### 2.2 Index Handling

**Rule**: By default, drop the DataFrame index. If keep_index=True, materialize
index levels as regular columns.

**Rationale**: Index semantics vary widely (meaningful vs positional). Default
behavior focuses on data content, not structural metadata.

**Implementation**:
- Default (keep_index=False): Call df.reset_index(drop=True)
- keep_index=True: Convert index to columns named '__index_level_0__',
  '__index_level_1__', etc., then reset index

**Examples**:
```python
# Original DataFrame with meaningful index
df = pd.DataFrame({'value': [1, 2]}, index=['A', 'B'])

# Default: index dropped
# Canonical columns: ['value']
# Rows: [[1], [2]]

# keep_index=True: index preserved as column
# Canonical columns: ['__index_level_0__', 'value']
# Rows: [['A', 1], ['B', 2]]
```

### 2.3 Column Ordering

**Rule**: Sort columns by their string representation in ascending order.

**Rationale**: Column order is often incidental to data meaning. Sorting ensures
consistent ordering regardless of DataFrame construction history.

**Implementation**:
```python
df = df.reindex(sorted(df.columns.astype(str)), axis=1)
```

**Edge Cases**:
- Non-string column names converted via str()
- Case-sensitive sorting (uppercase before lowercase)
- Special characters sorted by Unicode code point

**Examples**:
```python
# Original: columns ['z', 'a', 'B']
# Canonical: columns ['B', 'a', 'z']

# Mixed types: columns [1, 'a', 2.5]
# Canonical: columns ['1', '2.5', 'a']
```

### 2.4 Dtype Normalization

**Rule**: Convert all dtypes to a standardized, nullable set using pandas
convert_dtypes() as the base, with additional normalization rules.

**Rationale**: Different pandas versions and operations can produce different
but equivalent dtypes (int32 vs int64, object vs string). Normalization
ensures consistent representation.

#### 2.4.1 Integer Types
- **Target**: pandas nullable Int64Dtype where possible
- **Rationale**: Handles missing values consistently, avoids int32/int64 variations
- **Edge Cases**: Values exceeding Int64 range preserved as object dtype

#### 2.4.2 Float Types
- **Target**: float64 for all floating-point data
- **Representation**: Configurable via float_mode parameter
  - 'round-trip': Use Python repr() for exact representation
  - 'decimal=1e-N': Quantize to N decimal places using Decimal arithmetic
- **Rationale**: Eliminates float32/float64 differences, provides precision control
- **Special Values**:
  - NaN → ['null', None]
  - +inf → ['f', 'inf']
  - -inf → ['f', '-inf']

**Examples**:
```python
# round-trip mode
3.14159 → ['f', '3.14159']
1e-10 → ['f', '1e-10']

# decimal=1e-12 mode
3.14159265359 → ['f', '3.141592653590']
```

#### 2.4.3 String Types
- **Target**: pandas StringDtype (nullable)
- **Rationale**: Distinguishes strings from generic objects, handles nulls consistently

#### 2.4.4 Boolean Types
- **Target**: pandas BooleanDtype (nullable)
- **Rationale**: Handles missing boolean values (pd.NA) consistently

#### 2.4.5 Datetime Types
- **Target**: timezone-aware datetime64[ns] in specified timezone (default UTC)
- **Normalization Process**:
  1. Convert naive datetimes to UTC (assume UTC if no timezone)
  2. Convert timezone-aware datetimes to target timezone
  3. Format as ISO-8601 with microsecond precision
  4. Use 'Z' suffix for UTC, offset notation for others

**Rationale**: Eliminates timezone ambiguity while preserving temporal meaning.
Different timezone representations of the same moment produce identical fingerprints.

**Examples**:
```python
# Same moment in different timezones
'2024-01-01 00:00:00+00:00' → ['dt', '2024-01-01T00:00:00.000000Z']
'2023-12-31 16:00:00-08:00' → ['dt', '2024-01-01T00:00:00.000000Z']

# NaT handling
pd.NaT → ['null', None]
```

#### 2.4.6 Timedelta Types
- **Target**: Represent as total nanoseconds (integer)
- **Rationale**: Avoids ISO-8601 duration parsing complexity, ensures precision
- **Format**: ['td', nanoseconds_as_int]

**Examples**:
```python
pd.Timedelta('1 day') → ['td', 86400000000000]
pd.Timedelta('1.5 seconds') → ['td', 1500000000]
```

#### 2.4.7 Categorical Types
- **Normalization**: Sort categories alphabetically, encode values as strings
- **Rationale**: Category order is often arbitrary; sorting ensures consistency
- **Format**: ['cat', category_string_value]

**Examples**:
```python
# Original categories: ['red', 'blue', 'green']
# Normalized categories: ['blue', 'green', 'red']
# Value 'red' → ['cat', 'red']
# Missing category → ['null', None]
```

#### 2.4.8 Binary Data Types
- **Target**: Base64-encoded strings
- **Rationale**: Ensures text-safe representation in JSON
- **Format**: ['bytes', base64_string]

**Examples**:
```python
b'hello' → ['bytes', 'aGVsbG8=']
bytearray([1, 2, 3]) → ['bytes', 'AQID']
```

#### 2.4.9 Decimal Types
- **Target**: String representation preserving precision
- **Rationale**: Avoids float conversion precision loss
- **Format**: ['dec', decimal_string]

**Examples**:
```python
Decimal('3.14159265358979323846') → ['dec', '3.14159265358979323846']
```

### 2.5 Missing Value Normalization

**Rule**: All missing values (NaN, NaT, None, pd.NA) normalize to ['null', None].

**Rationale**: Different pandas operations can produce different missing value
representations. Unification ensures consistent fingerprints.

**Implementation**: Check pd.isna(value) for comprehensive missing value detection.

### 2.6 Value Encoding System

**Rule**: Each cell value is encoded as [type_tag, normalized_value] where
type_tag indicates the semantic type.

**Type Tags**:
- 'null': Missing values
- 'i': Integers
- 'f': Floats
- 's': Strings
- 'b': Booleans
- 'dt': Datetimes
- 'td': Timedeltas
- 'cat': Categorical values
- 'bytes': Binary data
- 'dec': Decimal numbers

**Rationale**: Type tags preserve semantic meaning during serialization,
enabling type-aware reconstruction if needed.

### 2.7 JSON Serialization

**Rule**: Serialize canonical document as deterministic JSON.

**Parameters**:
- sort_keys=True: Ensures consistent key ordering
- separators=(',', ':'): Minimal whitespace for consistency
- ensure_ascii=False: Allows Unicode characters

**Document Structure**:
```json
{
  "spec": "df-canon-v1",
  "columns": ["col1", "col2", ...],
  "rows": [
    [["type", value], ["type", value], ...],
    [["type", value], ["type", value], ...],
    ...
  ]
}
```

3. Edge Cases and Special Handling
----------------------------------

### 3.1 Large Integers
- Values exceeding Int64 range preserved as strings with 's' type tag
- Rationale: Maintains precision without overflow

### 3.2 Mixed-Type Columns
- Each value encoded according to its individual type
- Rationale: Preserves heterogeneous data semantics

### 3.3 Nested Objects
- Complex objects converted to string representation via str()
- Type tag: 's'
- Rationale: Provides deterministic fallback for unsupported types

### 3.4 Empty DataFrames
- Columns: empty list []
- Rows: empty list []
- Rationale: Consistent representation of zero-data case

### 3.5 Duplicate Column Names
- Handled by pandas' default behavior during reindexing
- May result in suffixed column names (.1, .2, etc.)

4. Configuration Options
------------------------

### 4.1 sort_columns (bool, default=True)
- Controls column sorting behavior
- False: Preserve original column order
- Use case: When column order is semantically meaningful

### 4.2 keep_index (bool, default=False)
- Controls index handling
- True: Materialize index as columns
- Use case: When index contains meaningful data

### 4.3 tz (str, default='UTC')
- Target timezone for datetime normalization
- Use case: Consistent timezone for multi-region data

### 4.4 float_mode (str, default='round-trip')
- Float representation strategy
- 'round-trip': Exact representation via repr()
- 'decimal=1e-N': Fixed precision quantization
- Use case: Balancing precision vs. noise tolerance

5. Implementation Requirements
------------------------------

### 5.1 Conformance
Implementations MUST:
- Follow all normalization rules exactly
- Produce identical JSON for identical logical data
- Support all specified configuration options
- Include spec version in output metadata

### 5.2 Error Handling
- Invalid configuration: Raise clear error messages
- Unsupported dtypes: Fall back to string representation
- Serialization failures: Propagate with context

### 5.3 Performance Considerations
- Minimize DataFrame copying during normalization
- Use vectorized operations where possible
- Consider memory usage for large DataFrames

6. Security Considerations
--------------------------

### 6.1 Hash Algorithm Selection
- Default: SHA-256 for cryptographic strength
- Alternative: BLAKE3 for performance (when available)
- Rationale: Collision resistance important for data integrity

### 6.2 Input Validation
- Validate configuration parameters
- Handle malicious or malformed data gracefully
- Avoid code injection through string representations

7. Future Extensions
--------------------

### 7.1 Planned Enhancements (v2+)
- CBOR canonical encoding (RFC 8949) as JSON alternative
- Additional hash algorithms (BLAKE3, SHA-3)
- Support for other DataFrame libraries (Polars, Dask)
- Streaming canonicalization for large datasets

### 7.2 Backward Compatibility Promise
Changes requiring new major versions:
- Modifications to normalization rules
- Changes to JSON document structure
- Alterations to type tag semantics
- Different default configuration values

8. References
-------------

- RFC 8949: Concise Binary Object Representation (CBOR)
- ISO 8601: Date and time format standard
- IEEE 754: Floating-point arithmetic standard
- pandas documentation: https://pandas.pydata.org/docs/
- JSON specification: RFC 7159

---

This specification defines df-canon v1. Implementations should reference this
document for authoritative canonicalization behavior.
"""
