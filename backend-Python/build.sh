#!/bin/bash
# build.sh

# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting Vercel-optimized build process..."

# Define a temporary directory for dependencies
TEMP_DEPS_DIR="/tmp/dependencies"
rm -rf "$TEMP_DEPS_DIR" # Clean up from previous builds
mkdir -p "$TEMP_DEPS_DIR"

# 1. Install dependencies into the temporary directory
echo "Installing dependencies from requirements.txt into $TEMP_DEPS_DIR..."
python3.9 -m pip install --no-cache-dir -t "$TEMP_DEPS_DIR" -r requirements.txt

# 2. Go into the temporary directory to perform slimming
cd "$TEMP_DEPS_DIR"

# 3. Remove unnecessary files to reduce size
echo "Initial dependency size: $(du -sh .)"
echo "Starting aggressive slimming of dependencies..."

# This is very aggressive. It can save a lot of space but might break packages
# that rely on this metadata at runtime. Test your application thoroughly.
echo "Removing package metadata (.dist-info, .egg-info)..."
find . -type d -name "*.dist-info" -exec rm -rf {} +
find . -type d -name "*.egg-info" -exec rm -rf {} +

# Strip shared object files (if any)
echo "Stripping binaries (.so files)..."
find . -type f -name "*.so" -exec strip {} \; 2>/dev/null || true

# Remove specific large, non-essential directories from data science packages
echo "Removing known large, non-essential directories from installed packages..."
rm -rf pandas/tests
rm -rf numpy/tests
rm -rf numpy/core/tests
rm -rf numpy/f2py

rm -rf scipy/tests
# WARNING: The following lines remove large parts of scipy.
# If your code uses these modules, you must comment out the corresponding line.
rm -rf scipy/fftpack
rm -rf scipy/linalg
rm -rf scipy/special
rm -rf scipy/integrate
rm -rf scipy/sparse

rm -rf numba/tests
rm -rf numba/targets
rm -rf numba/cext
rm -rf numba/core/runtime/_cache

# Remove caches, documentation, tests, and other non-essential files
echo "Removing general caches, docs, and tests..."
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type d -name "tests" -o -name "test" -o -name "doc" -o -name "docs" -o -name "examples" -exec rm -rf {} +
find . -type f \( -name "*.pyc" -o -name "*.pyi" -o -name "*.o" -o -name "*.a" -o -name "*.md" -o -name "*.rst" -o -name "*.txt" -o -name "*.html" -o -name "*.csv" \) -delete

echo "Final dependency size: $(du -sh .)"
echo "Slimming finished."

# 4. Copy slimmed dependencies back to the project root
echo "Copying slimmed dependencies back to project root..."
rsync -a --delete . "$OLDPWD/"

# Return to the original directory
cd "$OLDPWD"

echo "Vercel-optimized build process finished."
