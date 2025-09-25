#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Setting up development environment..."

# --- Sync deps ---
uv sync

# --- Install pre-commit hooks ---
uv run pre-commit install

echo "✅ Development environment ready!"
echo ""
echo "👉 Common commands:"
echo "   uv run pytest -v                # run tests"
echo "   uv run pre-commit run --all-files   # lint/format all files"
echo ""
echo "You're good to go 🎉"
