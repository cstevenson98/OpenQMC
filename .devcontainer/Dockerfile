# Use NVIDIA CUDA base image with Ubuntu 22.04
FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set environment variables for CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies and LLVM repository
RUN apt-get update && apt-get install -y \
    build-essential \
    gdb \
    git \
    wget \
    curl \
    python3 \
    python3-pip \
    python3-dev \
    pkg-config \
    libssl-dev \
    zlib1g-dev \
    ninja-build \
    ccache \
    vim \
    nano \
    sudo \
    software-properties-common \
    lsb-release \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install Clang 22 from LLVM repository
RUN wget https://apt.llvm.org/llvm.sh \
    && chmod +x llvm.sh \
    && ./llvm.sh 22 all \
    && rm llvm.sh

# Set Clang 22 as default compiler
RUN update-alternatives --install /usr/bin/cc cc /usr/bin/clang-22 100 \
    && update-alternatives --install /usr/bin/c++ c++ /usr/bin/clang++-22 100 \
    && update-alternatives --install /usr/bin/clang clang /usr/bin/clang-22 100 \
    && update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-22 100

# Install CMake 3.31+
RUN wget https://github.com/Kitware/CMake/releases/download/v3.31.2/cmake-3.31.2-linux-x86_64.sh \
    && chmod +x cmake-3.31.2-linux-x86_64.sh \
    && ./cmake-3.31.2-linux-x86_64.sh --skip-license --prefix=/usr/local \
    && rm cmake-3.31.2-linux-x86_64.sh

# Install Conan 2.x
RUN pip3 install --no-cache-dir conan

# Create a non-root user for development
RUN useradd -m -s /bin/bash -G sudo developer \
    && echo "developer ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set up Conan profile directory
USER developer
RUN conan profile detect --force

# Set working directory
WORKDIR /workspace

# Switch back to root for any final setup
USER root

# Keep container running
CMD ["/bin/bash"]
