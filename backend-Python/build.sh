#!/bin/bash
# build.sh

# 1. Upgrade pip
pip install --upgrade pip

# 2. Install dependencies
pip install -r requirements.txt

# 3. Find site-packages directory
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")

# 4. Go into site-packages
cd $SITE_PACKAGES

# 5. Remove unnecessary files to reduce size
echo "Initial size of site-packages: $(du -sh .)"

# Remove __pycache__ directories
find . -type d -name "__pycache__" -exec rm -r {} +

# Remove .pyc files
find . -type f -name "*.pyc" -delete

# Remove testing directories and files
find . -type d -name "tests" -exec rm -r {} +
find . -type d -name "test" -exec rm -r {} +

# Strip shared object files
find . -name "*.so" -type f -exec strip {} \;

# Remove specific large directories from libraries
rm -rf pandas/tests
rm -rf numpy/tests
rm -rf scipy/tests
rm -rf scipy/linalg/tests
rm -rf scipy/special/tests
rm -rf numba/tests

echo "Final size of site-packages: $(du -sh .)"
