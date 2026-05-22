#!/bin/bash
# build.sh - Automated build script for Underwater Restoration System

set -e  # Exit on error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Underwater Image Restoration - Build Script              ║${NC}"
echo -e "${BLUE}║   LibTorch + OpenCV + CUDA                                 ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$PROJECT_DIR/build"
LIBTORCH_PATH="${CMAKE_PREFIX_PATH%:*}"  # Get first path from CMAKE_PREFIX_PATH

ENABLE_CUDA=${ENABLE_CUDA:-ON}
BUILD_TRAINING=${BUILD_TRAINING:-ON}
BUILD_INFERENCE=${BUILD_INFERENCE:-ON}
BUILD_TYPE=${BUILD_TYPE:-Release}
NUM_JOBS=${NUM_JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[*]${NC} $1"
}

check_command() {
    if command -v "$1" &> /dev/null; then
        print_status "$1 found: $(command -v "$1")"
        return 0
    else
        print_error "$1 not found"
        return 1
    fi
}

# ============================================================================
# DEPENDENCY CHECKS
# ============================================================================

echo -e "${BLUE}[Dependencies Check]${NC}"
echo ""

MISSING_DEPS=0

# CMake
if ! check_command cmake; then
    MISSING_DEPS=1
fi

# C++ Compiler
if ! check_command g++ && ! check_command clang++ && ! check_command cl.exe; then
    print_error "No C++ compiler found"
    MISSING_DEPS=1
else
    if command -v g++ &> /dev/null; then
        CXX_VERSION=$(g++ --version | head -1)
    elif command -v clang++ &> /dev/null; then
        CXX_VERSION=$(clang++ --version | head -1)
    fi
    print_status "C++ Compiler: $CXX_VERSION"
fi

# OpenCV
if check_command pkg-config; then
    if pkg-config --exists opencv4; then
        OPENCV_VERSION=$(pkg-config --modversion opencv4)
        print_status "OpenCV $OPENCV_VERSION found"
    elif pkg-config --exists opencv; then
        OPENCV_VERSION=$(pkg-config --modversion opencv)
        print_status "OpenCV $OPENCV_VERSION found"
    else
        print_warning "OpenCV not found via pkg-config"
        print_info "Trying system default..."
    fi
fi

# LibTorch
if [ -n "$LIBTORCH_PATH" ] && [ -d "$LIBTORCH_PATH" ]; then
    print_status "LibTorch found: $LIBTORCH_PATH"
else
    print_warning "LibTorch not found in CMAKE_PREFIX_PATH"
    print_info "Download from: https://pytorch.org/get-started/locally/"
    read -p "Enter LibTorch path (or press Enter to skip): " LIBTORCH_PATH
    if [ -z "$LIBTORCH_PATH" ]; then
        print_error "LibTorch path required"
        MISSING_DEPS=1
    fi
fi

# CUDA
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
    print_status "CUDA $CUDA_VERSION found"
else
    if [ "$ENABLE_CUDA" = "ON" ]; then
        print_warning "CUDA not found. Falling back to CPU mode."
        ENABLE_CUDA=OFF
    fi
fi

echo ""

if [ $MISSING_DEPS -eq 1 ]; then
    print_error "Some required dependencies are missing!"
    echo ""
    echo "Installation guide:"
    echo "  Ubuntu/Debian:"
    echo "    sudo apt-get install -y build-essential cmake libopencv-dev"
    echo ""
    echo "  macOS:"
    echo "    brew install cmake opencv"
    echo ""
    exit 1
fi

# ============================================================================
# BUILD CONFIGURATION
# ============================================================================

echo -e "${BLUE}[Build Configuration]${NC}"
echo ""
print_info "Project: $PROJECT_DIR"
print_info "Build directory: $BUILD_DIR"
print_info "Build type: $BUILD_TYPE"
print_info "Parallel jobs: $NUM_JOBS"
print_info "CUDA enabled: $ENABLE_CUDA"
print_info "LibTorch: $LIBTORCH_PATH"
echo ""

# ============================================================================
# CREATE BUILD DIRECTORY
# ============================================================================

if [ ! -d "$BUILD_DIR" ]; then
    print_info "Creating build directory..."
    mkdir -p "$BUILD_DIR"
else
    print_info "Build directory already exists"
fi

cd "$BUILD_DIR"

# ============================================================================
# CMAKE CONFIGURATION
# ============================================================================

echo -e "${BLUE}[CMake Configuration]${NC}"
echo ""

CMAKE_ARGS=(
    "-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
    "-DENABLE_CUDA=$ENABLE_CUDA"
    "-DBUILD_TRAINING=$BUILD_TRAINING"
    "-DBUILD_INFERENCE=$BUILD_INFERENCE"
    "-DCMAKE_PREFIX_PATH=$LIBTORCH_PATH"
)

print_info "Running CMake..."
cmake "${CMAKE_ARGS[@]}" "$PROJECT_DIR"

if [ $? -ne 0 ]; then
    print_error "CMake configuration failed"
    exit 1
fi

print_status "CMake configuration successful"
echo ""

# ============================================================================
# BUILD
# ============================================================================

echo -e "${BLUE}[Building Project]${NC}"
echo ""

print_info "Starting build with $NUM_JOBS parallel jobs..."
echo ""

if ! cmake --build . -j "$NUM_JOBS" --config "$BUILD_TYPE"; then
    print_error "Build failed"
    exit 1
fi

print_status "Build completed successfully"
echo ""

# ============================================================================
# BUILD SUMMARY
# ============================================================================

echo -e "${BLUE}[Build Summary]${NC}"
echo ""

# Find built executables
if [ -f "./train_restoration" ]; then
    print_status "Training executable: $BUILD_DIR/train_restoration"
fi

if [ -f "./restore_images" ]; then
    print_status "Inference executable: $BUILD_DIR/restore_images"
fi

if [ -f "./Release/train_restoration.exe" ]; then
    print_status "Training executable: $BUILD_DIR/Release/train_restoration.exe"
fi

if [ -f "./Release/restore_images.exe" ]; then
    print_status "Inference executable: $BUILD_DIR/Release/restore_images.exe"
fi

echo ""

# ============================================================================
# NEXT STEPS
# ============================================================================

echo -e "${BLUE}[Next Steps]${NC}"
echo ""
echo "1. Prepare dataset:"
echo "   mkdir -p data/EUVP/{train,test}/{murky,reference}"
echo "   # Download EUVP dataset from http://li-lab.net/en/pub-page/euvp/"
echo ""
echo "2. Start training:"
echo "   cd $BUILD_DIR"
echo "   ./train_restoration"
echo ""
echo "3. Run inference:"
echo "   cd $BUILD_DIR"
echo "   ./restore_images ../input_images ../output_images"
echo ""
echo "4. View results:"
echo "   ls -lh $BUILD_DIR/../output_images/"
echo ""

print_status "Build complete! All systems go! 🚀"
