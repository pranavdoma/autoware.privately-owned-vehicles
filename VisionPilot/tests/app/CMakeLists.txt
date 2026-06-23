find_package(OpenCV REQUIRED)

add_executable(test_vision_pilot
        test_vision_pilot.cpp
)

target_link_libraries(test_vision_pilot
        PRIVATE
        config           # VisionPilotConfig, load_vision_pilot_config
        logging          # VP_INFO / VP_WARN / VP_ERROR
        engine           # OnnxEngine
        models           # AutoDrive, AutoSteer, AutoSpeed
        fusion           # LongitudinalFusion, LateralFusion
        planning
        vp_debug         # annotate_frame, DebugView
        vehicle_interface
        camera_interface
        image_preprocessing
        visualization
)