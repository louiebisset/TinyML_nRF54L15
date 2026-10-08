# 1. Required version (you just fixed this)
cmake_minimum_required(VERSION 3.21)

# 2. Pull in the Zephyr build system
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})

# 3. Name your project
project(dmic)

# 4. CRITICAL: Tell Zephyr where the code is
# Change "src/main.cpp" to "src/main.c" if you are using C
target_sources(app PRIVATE src/main.c)