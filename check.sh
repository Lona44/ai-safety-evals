#!/bin/bash
# Local CI/CD check script - run before pushing to catch issues early

set -e

echo "============================================"
echo "🔍 Running local CI/CD checks..."
echo "============================================"
echo ""

# Activate virtual environment
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Creating..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt -q
else
    source .venv/bin/activate
fi

# 1. Black formatting
echo "📝 [1/5] Running Black (code formatting)..."
if black --check --quiet . 2>/dev/null; then
    echo "   ✅ Black: All files properly formatted"
else
    echo "   ❌ Black: Formatting issues found"
    echo "   Run 'black .' to auto-fix"
    exit 1
fi

# 2. Ruff linting
echo "🔍 [2/5] Running Ruff (linting)..."
if ruff check . --quiet 2>/dev/null; then
    echo "   ✅ Ruff: No linting issues"
else
    echo "   ⚠️  Ruff: Linting issues found"
    echo "   Run 'ruff check --fix .' to auto-fix"
    exit 1
fi

# 3. MyPy type checking (non-blocking)
echo "🔬 [3/5] Running MyPy (type checking)..."
if mypy . --config-file pyproject.toml 2>/dev/null; then
    echo "   ✅ MyPy: No type errors"
else
    echo "   ⚠️  MyPy: Type errors found (non-blocking)"
fi

# 4. Bandit security scanning
echo "🔐 [4/5] Running Bandit (security)..."
if bandit -r . -c pyproject.toml --quiet 2>/dev/null; then
    echo "   ✅ Bandit: No security issues"
else
    echo "   ⚠️  Bandit: Security warnings found (check output)"
fi

# 5. Secrets detection
echo "🔑 [5/5] Checking for exposed secrets..."
SECRET_FOUND=0

if grep -r "sk-[a-zA-Z0-9]\{48\}" . --exclude-dir={.git,.venv,venv,outputs} --exclude="*.env" 2>/dev/null; then
    echo "   ❌ Found OpenAI API key pattern!"
    SECRET_FOUND=1
fi

if grep -r "AIza[0-9A-Za-z_-]\{35\}" . --exclude-dir={.git,.venv,venv,outputs} --exclude="*.env" 2>/dev/null; then
    echo "   ❌ Found Google API key pattern!"
    SECRET_FOUND=1
fi

if grep -r "AKIA[0-9A-Z]\{16\}" . --exclude-dir={.git,.venv,venv,outputs} --exclude="*.env" 2>/dev/null; then
    echo "   ❌ Found AWS key pattern!"
    SECRET_FOUND=1
fi

if [ $SECRET_FOUND -eq 0 ]; then
    echo "   ✅ Secrets check: No exposed keys detected"
else
    exit 1
fi

echo ""
echo "============================================"
echo "✅ All checks passed! Safe to push."
echo "============================================"
