cmake_minimum_required(VERSION 3.15)
project(UnderwaterRestoration LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# CONFIGURATION OPTIONS
# ============================================================================
option(ENABLE_CUDA "Enable CUDA support via LibTorch" ON)
option(BUILD_TRAINING "Build training executable" ON)
option(BUILD_INFERENCE "Build inference executable" ON)

# ============================================================================
# FIND PACKAGES
# ============================================================================

# LibTorch (PyTorch C++ API)
# Download from: https://pytorch.org/get-started/locally/
# Set CMAKE_PREFIX_PATH=/path/to/libtorch or use FindTorch module
find_package(Torch REQUIRED)
message(STATUS "Found Torch version: ${Torch_VERSION}")

# OpenCV
find_package(OpenCV 4.0 REQUIRED)
message(STATUS "Found OpenCV: ${OpenCV_VERSION}")

# ============================================================================
# INCLUDE DIRECTORIES
# ============================================================================
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})

# ============================================================================
# SOURCE FILES
# ============================================================================
set(SOURCES
    src/model.cpp
    src/dataloader.cpp
    src/loss_functions.cpp
)

set(HEADERS
    include/model.hpp
    include/dataloader.hpp
    include/loss_functions.hpp
    include/config.hpp
)

# ============================================================================
# LIBRARY TARGET
# ============================================================================
add_library(underwater_lib ${SOURCES} ${HEADERS})
target_link_libraries(underwater_lib
    PUBLIC
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBS}
)

# Set RPATH for LibTorch libraries
set_target_properties(underwater_lib PROPERTIES
    BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
    INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
)

# ============================================================================
# TRAINING EXECUTABLE
# ============================================================================
if(BUILD_TRAINING)
    add_executable(train_restoration src/train.cpp)
    target_link_libraries(train_restoration
        PRIVATE
            underwater_lib
            ${TORCH_LIBRARIES}
            ${OpenCV_LIBS}
    )
    set_target_properties(train_restoration PROPERTIES
        BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
        INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
    )
endif()

# ============================================================================
# INFERENCE EXECUTABLE
# ============================================================================
if(BUILD_INFERENCE)
    add_executable(restore_images src/inference.cpp)
    target_link_libraries(restore_images
        PRIVATE
            underwater_lib
            ${TORCH_LIBRARIES}
            ${OpenCV_LIBS}
    )
    set_target_properties(restore_images PROPERTIES
        BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
        INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
    )
endif()

# ============================================================================
# EVALUATION EXECUTABLE
# ============================================================================
option(BUILD_EVALUATION "Build evaluation executable" ON)
if(BUILD_EVALUATION)
    add_executable(evaluate src/evaluate.cpp)
    target_link_libraries(evaluate
        PRIVATE
            underwater_lib
            ${TORCH_LIBRARIES}
            ${OpenCV_LIBS}
    )
    set_target_properties(evaluate PROPERTIES
        BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
        INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
    )
endif()

# ============================================================================
# VISUALIZATION EXECUTABLE
# ============================================================================
option(BUILD_VISUALIZATION "Build visualization executable" ON)
if(BUILD_VISUALIZATION)
    add_executable(visualize src/visualize.cpp)
    target_link_libraries(visualize
        PRIVATE
            underwater_lib
            ${TORCH_LIBRARIES}
            ${OpenCV_LIBS}
    )
    set_target_properties(visualize PROPERTIES
        BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
        INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
    )
endif()

# ============================================================================
# COMPILER FLAGS
# ============================================================================
if(MSVC)
    target_compile_options(underwater_lib PRIVATE /W4 /O2)
else()
    target_compile_options(underwater_lib PRIVATE -Wall -Wextra -O3)
    if(ENABLE_CUDA)
        target_compile_options(underwater_lib PRIVATE -march=native)
    endif()
endif()

# ============================================================================
# PRINT BUILD SUMMARY
# ============================================================================
message(STATUS "")
message(STATUS "=== Build Configuration ===")
message(STATUS "CUDA Support: ${ENABLE_CUDA}")
message(STATUS "Build Training: ${BUILD_TRAINING}")
message(STATUS "Build Inference: ${BUILD_INFERENCE}")
message(STATUS "Build Evaluation: ${BUILD_EVALUATION}")
message(STATUS "Build Visualization: ${BUILD_VISUALIZATION}")
message(STATUS "LibTorch: ${TORCH_INSTALL_PREFIX}")
message(STATUS "OpenCV: ${OpenCV_VERSION}")
message(STATUS "")
message(STATUS "Targets created:")
if(BUILD_TRAINING)
    message(STATUS "  - train_restoration")
endif()
if(BUILD_INFERENCE)
    message(STATUS "  - restore_images")
endif()
if(BUILD_EVALUATION)
    message(STATUS "  - evaluate")
endif()
if(BUILD_VISUALIZATION)
    message(STATUS "  - visualize")
endif()
message(STATUS "")
