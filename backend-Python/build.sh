#!/bin/bash
# build.sh

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting aggressive build process..."

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
echo "Starting aggressive slimming..."

# Remove __pycache__ directories and .pyc files
echo "Removing caches..."
find . -type d -name "__pycache__" -exec rm -r {} +
find . -type f -name "*.pyc" -delete

# Remove testing directories
echo "Removing test directories..."
find . -type d -name "tests" -exec rm -r {} +
find . -type d -name "test" -exec rm -r {} +

# Remove metadata directories - this is aggressive but can save a lot of space
echo "Removing metadata..."
find . -type d -name "*.dist-info" -exec rm -r {} +
find . -type d -name "*.egg-info" -exec rm -r {} +

# Strip shared object files (if any)
echo "Stripping binaries..."
find . -name "*.so" -type f -exec strip {} \;

# Remove specific large directories from libraries that are known to be non-essential
echo "Removing known large, non-essential directories..."
rm -rf pandas/tests
rm -rf numpy/tests
rm -rf scipy/tests
rm -rf scipy/linalg/tests
rm -rf scipy/special/tests
rm -rf numba/tests

# Remove documentation and other text files
find . -type f -name "*.md" -delete
find . -type f -name "*.txt" -delete

echo "Final size of site-packages: $(du -sh .)"
echo "Aggressive build process finished."
