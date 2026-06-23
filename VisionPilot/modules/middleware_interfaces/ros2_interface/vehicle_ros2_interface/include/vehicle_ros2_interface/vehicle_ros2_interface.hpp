#ifndef VISIONPILOT_VEHICLE_ROS2_INTERFACE_HPP
#define VISIONPILOT_VEHICLE_ROS2_INTERFACE_HPP

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64.hpp>
#include <thread>
#include <vehicle_interface/vehicle_interface.hpp>

class VehicleRos2Interface : public VehicleInterface
{
public:
    VehicleRos2Interface();
    ~VehicleRos2Interface() override;

    double read() override;
    void write(double steering, double acceleration) override;

private:
    // Inner node — owns all ROS 2 concerns
    class VehicleRos2Node : public rclcpp::Node
    {
    public:
        explicit VehicleRos2Node();
        ~VehicleRos2Node() = default;

        rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr sub_;
        rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_;
    };

    std::shared_ptr<VehicleRos2Node> node_;
    rclcpp::executors::SingleThreadedExecutor executor_;
    std::thread spin_thread_;
};

#endif //VISIONPILOT_VEHICLE_ROS2_INTERFACE_HPP