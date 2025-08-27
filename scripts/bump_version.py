#!/usr/bin/env python3
"""
Version bumping script for df-fingerprint.

Usage:
    python scripts/bump_version.py patch  # 0.1.0 -> 0.1.1
    python scripts/bump_version.py minor  # 0.1.0 -> 0.2.0  
    python scripts/bump_version.py major  # 0.1.0 -> 1.0.0
"""

import re
import sys
from pathlib import Path


def get_current_version():
    """Get current version from _version.py"""
    version_file = Path("df_fingerprint/_version.py")
    content = version_file.read_text()
    match = re.search(r'__version__ = "([^"]+)"', content)
    if not match:
        raise ValueError("Could not find version in _version.py")
    return match.group(1)


def bump_version(current_version, bump_type):
    """Bump version according to semver rules"""
    major, minor, patch = map(int, current_version.split('.'))
    
    if bump_type == 'major':
        return f"{major + 1}.0.0"
    elif bump_type == 'minor':
        return f"{major}.{minor + 1}.0"
    elif bump_type == 'patch':
        return f"{major}.{minor}.{patch + 1}"
    else:
        raise ValueError(f"Invalid bump type: {bump_type}")


def update_version_in_file(file_path, old_version, new_version):
    """Update version in a file"""
    content = Path(file_path).read_text()
    updated = content.replace(f'"{old_version}"', f'"{new_version}"')
    Path(file_path).write_text(updated)
    print(f"Updated {file_path}: {old_version} -> {new_version}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/bump_version.py [major|minor|patch]")
        sys.exit(1)
    
    bump_type = sys.argv[1]
    if bump_type not in ['major', 'minor', 'patch']:
        print("Bump type must be one of: major, minor, patch")
        sys.exit(1)
    
    # Get current version
    current_version = get_current_version()
    new_version = bump_version(current_version, bump_type)
    
    print(f"Bumping version: {current_version} -> {new_version}")
    
    # Update version in files
    update_version_in_file("df_fingerprint/_version.py", current_version, new_version)
    update_version_in_file("pyproject.toml", current_version, new_version)
    
    print(f"\nVersion bumped to {new_version}")
    print("Next steps:")
    print("1. Update CHANGELOG.md with new version")
    print("2. Commit changes: git add -A && git commit -m 'Bump version to {new_version}'")
    print("3. Create tag: git tag v{new_version}")
    print("4. Push: git push && git push --tags")


if __name__ == "__main__":
    main()