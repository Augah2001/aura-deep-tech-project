#!/bin/bash
# build.sh

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting Vercel-optimized build process..."

# Define the output directory for Vercel
OUTPUT_DIR=".vercel/output/src"
mkdir -p "$OUTPUT_DIR"

# 1. Install dependencies into the output directory
echo "Installing dependencies from requirements.txt into $OUTPUT_DIR..."
python3.9 -m pip install -t "$OUTPUT_DIR" -r requirements.txt

# 2. Copy application files into the output directory
echo "Copying application files (main.py, app/) into $OUTPUT_DIR..."
cp main.py "$OUTPUT_DIR/"
cp -r app "$OUTPUT_DIR/"

# 3. Go into the output directory to perform slimming
cd "$OUTPUT_DIR"

# 4. Remove unnecessary files to reduce size
echo "Initial size of $OUTPUT_DIR: $(du -sh .)"
echo "Starting aggressive slimming of dependencies..."

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

echo "Final size of $OUTPUT_DIR: $(du -sh .)"
echo "Vercel-optimized build process finished."

