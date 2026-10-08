cmake_minimum_required(VERSION 3.20.0)

set(ZEPHYR_EXTRA_MODULES "C:/ncs/v3.1.1/modules/lib/tflite-micro")

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(Shapes)

target_sources(app PRIVATE src/main.cpp)

# TFLite Includes
set(TFLITE_ROOT "C:/ncs/v3.1.1/modules/lib/tflite-micro")
target_include_directories(app PRIVATE ${TFLITE_ROOT})
target_include_directories(app PRIVATE ${TFLITE_ROOT}/third_party/flatbuffers/include)
target_include_directories(app PRIVATE ${TFLITE_ROOT}/third_party/gemmlowp)
target_include_directories(app PRIVATE ${TFLITE_ROOT}/third_party/ruy)