#!/usr/bin/env bash
set -euo pipefail

# Mirror the GitHub Actions CI test matrix locally via tox.
python -m pip install --upgrade pip
python -m pip install tox
python -m tox
