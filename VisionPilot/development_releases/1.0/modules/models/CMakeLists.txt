cmake_minimum_required(VERSION 3.22.1)

add_library(models
    src/auto_drive.cpp
    src/auto_steer.cpp
    src/auto_speed.cpp
)

target_include_directories(models
    PUBLIC
        include
)

# engine provides OnnxEngine + ORT headers/libs transitively
target_link_libraries(models
    PUBLIC
        engine
)

target_compile_features(models PUBLIC cxx_std_17)
