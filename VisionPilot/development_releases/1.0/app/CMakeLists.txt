add_executable(VisionPilot vision_pilot.cpp)

target_link_libraries(VisionPilot
        PRIVATE
        can_interface
        v4l2_interface
        models
        visualization
        rclcpp::rclcpp
        sensor_msgs::sensor_msgs
        OpenCV::OpenCV
        cv_bridge::cv_bridge
)