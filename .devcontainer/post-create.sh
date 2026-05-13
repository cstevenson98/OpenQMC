#!/bin/bash

# Post-create script for dev container setup

echo "=== OpenQMC Dev Container Setup ==="

# Set up Conan profile
echo "Setting up Conan profile..."
conan profile detect --force || true

# Create build directory
echo "Creating build directory..."
mkdir -p /workspace/build

echo ""
echo "=== Running GPU Verification ==="
bash /workspace/.devcontainer/verify-gpu.sh

if [ $? -eq 0 ]; then
    echo ""
    echo "=== Setup Complete ==="
    echo ""
    echo "You can now build the project with:"
    echo "  conan install . --output-folder=build --build=missing"
    echo "  cd build"
    echo "  cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release"
    echo "  cmake --build . -j\$(nproc)"
else
    echo ""
    echo "Warning: GPU verification failed. Check your NVIDIA Container Toolkit setup."
fi
