#!/bin/bash
# build.sh

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting Vercel-optimized build process..."

# Define a temporary directory for dependencies
TEMP_DEPS_DIR="/tmp/dependencies"
mkdir -p "$TEMP_DEPS_DIR"

# 1. Install dependencies into the temporary directory
echo "Installing dependencies from requirements.txt into $TEMP_DEPS_DIR..."
python3.9 -m pip install -t "$TEMP_DEPS_DIR" -r requirements.txt
python3.9 -m pip install --no-cache-dir -t "$TEMP_DEPS_DIR" -r requirements.txt

# 2. Go into the temporary directory to perform slimming
cd "$TEMP_DEPS_DIR"

# 3. Remove unnecessary files to reduce size
echo "Initial size of $TEMP_DEPS_DIR: $(du -sh .)"
echo "Starting aggressive slimming of dependencies..."
echo "Starting slimming of dependencies..."

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

# Remove testing, documentation, and other non-essential files more broadly
echo "Removing tests, docs, and other non-essential files..."
find . -type d -name "tests" -exec rm -rf {} +
find . -type d -name "test" -exec rm -rf {} +
find . -type d -name "doc" -exec rm -rf {} +
find . -type d -name "docs" -exec rm -rf {} +
find . -type f -name "*.md" -delete
find . -type f -name "*.txt" -delete

# Strip shared object files (if any)
echo "Stripping binaries..."
find . -name "*.so" -type f -exec strip {} \;
find . -name "*.so" -type f -exec strip {} \; 2>/dev/null || true

# Remove specific large directories from libraries that are known to be non-essential
echo "Removing known large, non-essential directories..."
rm -rf pandas/tests
rm -rf numpy/tests
rm -rf scipy/tests
rm -rf scipy/linalg/tests
rm -rf scipy/special/tests
rm -rf numba/tests
# Add more specific removals if needed

# More aggressive slimming:
# Remove documentation, examples, and other non-essential files
find . -type d -name "doc" -exec rm -r {} +
find . -type d -name "docs" -exec rm -r {} +
find . -type d -name "examples" -exec rm -r {} +
find . -type d -name "locale" -exec rm -r {} +
find . -type f -name "*.md" -delete
find . -type f -name "*.txt" -delete
find . -type f -name "*.rst" -delete
find . -type f -name "*.chm" -delete
find . -type f -name "*.html" -delete

# Remove command-line scripts that are not needed for a serverless function
find . -type f -name "*-script.py" -delete

echo "Final size of $TEMP_DEPS_DIR: $(du -sh .)"
echo "Aggressive slimming finished."
echo "Slimming finished."

# 4. Copy slimmed dependencies back to the project root
echo "Copying slimmed dependencies back to project root..."
cp -r ./* "$OLDPWD/"
rsync -a --delete . "$OLDPWD/"

# Return to the original directory
cd "$OLDPWD"

echo "Vercel-optimized build process finished."
