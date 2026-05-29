add_executable(VisionPilot
    vision_pilot.cpp
)

target_link_libraries(VisionPilot
    PRIVATE
    config           # VisionPilotConfig, load_vision_pilot_config
    logging          # VP_INFO / VP_WARN / VP_ERROR
    engine           # OnnxEngine
    models           # AutoDrive, AutoSteer, AutoSpeed
    fusion           # LongitudinalFusion, LateralFusion
    vp_debug         # annotate_frame, DebugView
    can_interface
    v4l2_interface
    camera_subscriber
    visualization
    rclcpp::rclcpp
)

# Suppress unresolved-symbol errors from system shared libs (gdal→curl, ICU from miniforge3)
# that are pulled in transitively by opencv_viz; we never use viz ourselves.
target_link_options(VisionPilot PRIVATE -Wl,--allow-shlib-undefined)
