# OpenQMC Dev Container

This dev container provides a complete CUDA development environment with GPU access for the OpenQMC project.

## Prerequisites

### 1. Install Docker
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

### 2. Install NVIDIA Container Toolkit

```bash
# Add the package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-container-toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use the NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 3. Verify GPU Access

Test that Docker can access your GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.3.1-base-ubuntu22.04 nvidia-smi
```

You should see your GPU information displayed.

## Using the Dev Container

### Option 1: VS Code (Recommended)

1. Install the "Dev Containers" extension in VS Code
2. Open this project folder in VS Code
3. Press `F1` and select "Dev Containers: Reopen in Container"
4. Wait for the container to build and start

### Option 2: Command Line

```bash
# Build the container
docker build -t openqmc-dev .devcontainer

# Run the container with GPU access
docker run --rm -it \
  --gpus all \
  --runtime=nvidia \
  -v $(pwd):/workspace \
  openqmc-dev
```

## Building the Project

Once inside the dev container:

```bash
# Install dependencies with Conan
conan install . --output-folder=build --build=missing

# Configure with CMake
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . -j$(nproc)

# Run tests
ctest --output-on-failure

# Run the main executable
./main
```

## Verifying GPU Access

Inside the container, run:

```bash
nvidia-smi
nvcc --version
```

## Features

- **CUDA 12.3.1** with full development tools
- **CMake 3.31+** for modern C++ builds
- **Conan 2.x** for dependency management
- **GCC 13** with C++23 support
- **Python 3** for helper scripts
- **VS Code extensions** for C++, CMake, and CUDA development
- **GPU access** via NVIDIA Container Toolkit

## Troubleshooting

### GPU not accessible

If `nvidia-smi` doesn't work inside the container:

1. Verify the NVIDIA Container Toolkit is installed on the host
2. Check that your Docker daemon is configured to use the NVIDIA runtime
3. Ensure you have the latest NVIDIA drivers installed on your host system

### Permission issues

If you encounter permission issues with files:

```bash
sudo chown -R developer:developer /workspace
```

### Conan profile issues

If Conan can't detect your profile:

```bash
conan profile detect --force
```
