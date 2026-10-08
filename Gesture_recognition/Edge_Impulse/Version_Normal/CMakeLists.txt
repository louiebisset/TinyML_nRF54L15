cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})

project(mic_edge_batt)

# 1. Edge Impulse SDK directory
set(EI_SDK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/edge-impulse-sdk)

# 2. Gather all Edge Impulse source files
FILE(GLOB_RECURSE EI_SOURCES
    "src/edge-impulse-sdk/*.cpp"
    "src/edge-impulse-sdk/*.c"
    "src/edge-impulse-sdk/*.cc"
    "src/model-parameters/*.cpp"
    "src/tflite-model/*.cpp"
)

# 3. All sources
target_sources(app PRIVATE
    src/main.cpp
    ${EI_SOURCES}
)

# 4. Include directories
target_include_directories(app PRIVATE
    src
    src/edge-impulse-sdk
    src/edge-impulse-sdk/classifier
    src/edge-impulse-sdk/porting
    src/edge-impulse-sdk/dsp
    src/edge-impulse-sdk/dsp/numpy_types
    src/edge-impulse-sdk/tensorflow
    src/edge-impulse-sdk/third_party
    src/edge-impulse-sdk/third_party/flatbuffers/include
    src/edge-impulse-sdk/third_party/gemmlowp
    src/edge-impulse-sdk/third_party/ruy
    src/model-parameters
    src/tflite-model
)

# 5. Compile definitions
target_compile_definitions(app PRIVATE
    EI_CLASSIFIER_TFLITE_ENABLE_CMSIS_NN=1
)

target_link_libraries(app PUBLIC -lm)