#!/bin/bash

echo "======================================"
echo "OpenQMC GPU Verification Script"
echo "======================================"
echo ""

# Check NVIDIA driver
echo "1. Checking NVIDIA Driver..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo "✓ NVIDIA driver accessible"
else
    echo "✗ NVIDIA driver NOT accessible"
    echo "  Make sure NVIDIA Container Toolkit is installed on the host"
    exit 1
fi

echo ""
echo "======================================"
echo ""

# Check CUDA compiler
echo "2. Checking CUDA Compiler..."
if command -v nvcc &> /dev/null; then
    nvcc --version
    echo "✓ NVCC compiler available"
else
    echo "✗ NVCC compiler NOT found"
    exit 1
fi

echo ""
echo "======================================"
echo ""

# Check CMake
echo "3. Checking CMake..."
if command -v cmake &> /dev/null; then
    cmake --version | head -n 1
    echo "✓ CMake available"
else
    echo "✗ CMake NOT found"
    exit 1
fi

echo ""
echo "======================================"
echo ""

# Check Conan
echo "4. Checking Conan..."
if command -v conan &> /dev/null; then
    conan --version
    echo "✓ Conan available"
else
    echo "✗ Conan NOT found"
    exit 1
fi

echo ""
echo "======================================"
echo ""

# Compile a simple CUDA test
echo "5. Testing CUDA Compilation..."
cat > /tmp/cuda_test.cu << 'EOF'
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void hello_cuda() {
    printf("Hello from GPU thread %d!\n", threadIdx.x);
}

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    printf("Found %d CUDA device(s)\n", deviceCount);
    
    if (deviceCount > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("Device 0: %s\n", prop.name);
        printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
        
        hello_cuda<<<1, 5>>>();
        cudaDeviceSynchronize();
        
        printf("✓ CUDA test passed!\n");
        return 0;
    } else {
        printf("✗ No CUDA devices found\n");
        return 1;
    }
}
EOF

nvcc /tmp/cuda_test.cu -o /tmp/cuda_test
if [ $? -eq 0 ]; then
    echo "✓ CUDA compilation successful"
    echo ""
    echo "6. Running CUDA Test Program..."
    /tmp/cuda_test
    TEST_RESULT=$?
    rm -f /tmp/cuda_test /tmp/cuda_test.cu
    
    if [ $TEST_RESULT -eq 0 ]; then
        echo ""
        echo "======================================"
        echo "✓ All checks passed!"
        echo "Your dev container is ready for CUDA development."
        echo "======================================"
        exit 0
    else
        echo "✗ CUDA test program failed"
        exit 1
    fi
else
    echo "✗ CUDA compilation failed"
    rm -f /tmp/cuda_test.cu
    exit 1
fi
