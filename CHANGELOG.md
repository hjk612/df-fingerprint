# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-01-XX

### Added
- Initial release of df-fingerprint
- Core fingerprinting functionality with `fingerprint()` function
- `Canonicalizer` class for configurable DataFrame canonicalization
- Support for df-canon-v1 specification
- Vectorized column-wise encoding for high performance
- Comprehensive test suite with property-based testing
- Support for SHA-256 and BLAKE3 hash algorithms
- Cross-environment stability guarantees
- Extensive documentation and examples

### Features
- **Performance**: 150k+ rows/second on mixed-type DataFrames
- **Compatibility**: Works across Python 3.9+, pandas 1.5+
- **Deterministic**: Same logical data → same fingerprint
- **Configurable**: Sort columns, timezone handling, float precision
- **Robust**: Handles missing values, categoricals, timezones, edge cases

### Specification
- Implements df-canon-v1 canonicalization specification
- JSON-based canonical representation
- Deterministic serialization with sorted keys
- Type-tagged encoding for all pandas dtypes
- UTC timezone normalization for datetimes
- Missing value normalization (NaN/None/NaT/pd.NA → null)

[Unreleased]: https://github.com/hjk612/df-fingerprint/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/hjk612/df-fingerprint/releases/tag/v0.1.0