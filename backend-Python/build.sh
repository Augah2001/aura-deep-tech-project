#!/bin/bash
# build.sh

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting build process..."

# 1. Install dependencies
echo "Installing dependencies from requirements.txt..."
python3.9 -m pip install -r requirements.txt

# 2. Find site-packages directory
echo "Locating site-packages directory..."
SITE_PACKAGES=$(python3.9 -c "import site; print(site.getsitepackages()[0])")
echo "Found site-packages at: $SITE_PACKAGES"

# 3. Go into site-packages
cd "$SITE_PACKAGES"

# 4. Remove unnecessary files to reduce size
echo "Initial size of site-packages: $(du -sh .)"
echo "Slimming down packages..."

# Remove __pycache__ directories and .pyc files
find . -type d -name "__pycache__" -exec rm -r {} +
find . -type f -name "*.pyc" -delete

# Remove testing directories
find . -type d -name "tests" -exec rm -r {} +
find . -type d -name "test" -exec rm -r {} +

# Strip shared object files (if any)
find . -name "*.so" -type f -exec strip {} \;

# Remove specific large directories from libraries
rm -rf pandas/tests
rm -rf numpy/tests
rm -rf scipy/tests
rm -rf scipy/linalg/tests
rm -rf scipy/special/tests
rm -rf numba/tests

echo "Final size of site-packages: $(du -sh .)"
echo "Build process finished."