# Migration Guide: v1.x to v2.0

## Overview

Version 2.0 introduces significant changes to the build system and development workflow. This document outlines the key changes and migration steps.

## Key Changes

### 1. Package Manager: pip/flit → uv

**Why?** uv is significantly faster (10-100x) than pip and provides better dependency resolution.

**Before (v1.x):**
```bash
pip install -e ".[dev]"
```

**After (v2.0):**
```bash
uv sync --extra dev
```

### 2. Build Backend: flit → hatchling

**Why?** Hatchling is uv's recommended build backend and is more actively maintained.

- More consistent with modern Python packaging standards
- Better integration with uv
- Simpler configuration

### 3. Linter/Formatter: black + isort → ruff

**Why?** Ruff is 10-100x faster and combines multiple tools into one.

**Before (v1.x):**
```bash
black src/
isort src/
```

**After (v2.0):**
```bash
uv run ruff format src/
uv run ruff check --fix src/
```

### 4. Testing: Added pytest

Version 2.0 includes pytest for testing (previously no test framework was configured).

```bash
uv run pytest
```

### 5. Dependency Changes

**Added:**
- `ruff` - Linting and formatting
- `pytest` + `pytest-cov` - Testing
- `jupyter` + `ipykernel` - Notebook support (dev)

**Updated:**
- `numpy`: Now pinned to `>=2.0.0,<3.0.0` (more specific)
- `pandas`: Updated to `>=2.2.2`
- `magma-var`: Now using `>=0.0.9` (corrected from non-existent 0.1.0)

### 6. Python Version Support

**Before:** `>=3.11` (open-ended)
**After:** `>=3.11,<3.14` (explicit upper bound)

This prevents compatibility issues with unreleased Python versions.

## Migration Steps for Developers

### 1. Install uv

```bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and Setup

```bash
# Pull the latest changes
git checkout dev-2.0.0
git pull

# Remove old virtual environment (if exists)
rm -rf .venv venv

# Sync dependencies
uv sync --extra dev

# Activate environment
# Windows:
.venv\Scripts\activate
# Unix/Mac:
source .venv/bin/activate
```

### 3. Update Your Workflow

Replace old commands with new ones:

| Old Command | New Command |
|-------------|-------------|
| `pip install -e .` | `uv sync` |
| `pip install package` | `uv add package` |
| `black src/` | `uv run ruff format src/` |
| `isort src/` | `uv run ruff check --fix src/` |
| `python -m pytest` | `uv run pytest` |

## Files Changed

### Modified
- `pyproject.toml` - Updated build system, dependencies, and tool configs
- `CLAUDE.md` - Updated development commands

### Added
- `uv.lock` - Dependency lock file (commit this to git)
- `MIGRATION_V2.md` - This file

### Removed
- None (backward compatibility maintained where possible)

## Breaking Changes

⚠️ **For Developers:**

- The virtual environment must be recreated with `uv sync`
- Old `.venv` created with pip will not work
- You must install `uv` to work on v2.0+

## Rollback

If you need to rollback to v1.x:

```bash
git checkout master
rm -rf .venv
pip install -e ".[dev]"
```

## Questions?

Open an issue on GitHub: https://github.com/martanto/magma-multigas/issues
