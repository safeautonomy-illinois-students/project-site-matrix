#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <px4_msgs/msg/vehicle_status.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

using namespace std::chrono_literals;

class MpcOptimalPlannerCpp : public rclcpp::Node {
public:
  MpcOptimalPlannerCpp()
  : Node("mpc_optimal_uav_cpp_node") {
    enable_depth_camera_ = this->declare_parameter<bool>("enable_depth_camera", true);
    publish_period_sec_ = this->declare_parameter<double>("publish_period_sec", 0.05);
    offboard_period_sec_ = this->declare_parameter<double>("offboard_heartbeat_period_sec", 0.05);
    goal_ned_ = {
      static_cast<float>(this->declare_parameter<double>("goal_x", 0.0)),
      static_cast<float>(this->declare_parameter<double>("goal_y", 20.0)),
      static_cast<float>(this->declare_parameter<double>("goal_z", -3.0)),
    };
    latest_pos_sp_ned_ = goal_ned_;
    latest_vel_sp_ned_ = {0.0f, 0.0f, 0.0f};

    auto px4_qos = rclcpp::QoS(rclcpp::KeepLast(1));
    px4_qos.best_effort();
    px4_qos.transient_local();

    auto sensor_qos = rclcpp::QoS(rclcpp::KeepLast(1));
    sensor_qos.best_effort();

    if (enable_depth_camera_) {
      depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/depth_camera",
        sensor_qos,
        std::bind(&MpcOptimalPlannerCpp::on_depth, this, std::placeholders::_1));
    }

    odom_sub_ = this->create_subscription<px4_msgs::msg::VehicleOdometry>(
      "/fmu/out/vehicle_odometry",
      px4_qos,
      std::bind(&MpcOptimalPlannerCpp::on_odometry, this, std::placeholders::_1));
    status_sub_ = this->create_subscription<px4_msgs::msg::VehicleStatus>(
      "/fmu/out/vehicle_status",
      px4_qos,
      std::bind(&MpcOptimalPlannerCpp::on_status, this, std::placeholders::_1));

    sp_pub_ = this->create_publisher<px4_msgs::msg::TrajectorySetpoint>(
      "/fmu/in/trajectory_setpoint",
      px4_qos);
    offboard_pub_ = this->create_publisher<px4_msgs::msg::OffboardControlMode>(
      "/fmu/in/offboard_control_mode",
      px4_qos);
    mpc_label_path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/mpc/trajectory_sequence", 10);
    mpc_traj_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/mpc/trajectory", 10);

    const auto publish_period = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>(publish_period_sec_));
    const auto offboard_period = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>(offboard_period_sec_));

    publish_timer_ = this->create_wall_timer(
      publish_period,
      std::bind(&MpcOptimalPlannerCpp::publish_setpoint, this));
    offboard_timer_ = this->create_wall_timer(
      offboard_period,
      std::bind(&MpcOptimalPlannerCpp::publish_offboard_heartbeat, this));
    solver_timer_ = this->create_wall_timer(
      100ms,
      std::bind(&MpcOptimalPlannerCpp::run_solver_tick, this));

    if (enable_depth_camera_) {
      RCLCPP_INFO(this->get_logger(), "Depth camera enabled for C++ MPC scaffold.");
    } else {
      RCLCPP_WARN(this->get_logger(), "Depth camera disabled for C++ MPC scaffold.");
    }
    RCLCPP_WARN(
      this->get_logger(),
      "C++ MPC scaffold active. Port the Python hot path into run_solver_tick().");
  }

private:
  static std::array<float, 3> nan3() {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    return {nan, nan, nan};
  }

  uint64_t now_us() {
    return static_cast<uint64_t>(this->get_clock()->now().nanoseconds() / 1000);
  }

  void on_depth(const sensor_msgs::msg::Image::SharedPtr /*msg*/) {
    ++depth_frame_count_;
  }

  void on_odometry(const px4_msgs::msg::VehicleOdometry::SharedPtr msg) {
    current_pos_ned_ = {msg->position[0], msg->position[1], msg->position[2]};
    current_vel_ned_ = {msg->velocity[0], msg->velocity[1], msg->velocity[2]};
  }

  void on_status(const px4_msgs::msg::VehicleStatus::SharedPtr msg) {
    nav_state_ = static_cast<int>(msg->nav_state);
  }

  void run_solver_tick() {
    // Placeholder scaffold: hold the vehicle while the MPC core is ported from Python.
    latest_pos_sp_ned_ = current_pos_ned_;
    latest_vel_sp_ned_ = {0.0f, 0.0f, 0.0f};

    nav_msgs::msg::Path path_msg;
    path_msg.header.stamp = this->now();
    path_msg.header.frame_id = "ned";
    mpc_label_path_pub_->publish(path_msg);

    geometry_msgs::msg::PoseStamped pose_msg;
    pose_msg.header.stamp = path_msg.header.stamp;
    pose_msg.header.frame_id = "ned";
    pose_msg.pose.position.x = static_cast<double>(current_pos_ned_[0]);
    pose_msg.pose.position.y = static_cast<double>(current_pos_ned_[1]);
    pose_msg.pose.position.z = static_cast<double>(current_pos_ned_[2]);
    pose_msg.pose.orientation.w = 1.0;
    mpc_traj_pub_->publish(pose_msg);

    RCLCPP_WARN_THROTTLE(
      this->get_logger(),
      *this->get_clock(),
      5000,
      "C++ MPC scaffold is publishing a hold command. Port the Python solve core next.");
  }

  void publish_setpoint() {
    px4_msgs::msg::TrajectorySetpoint sp{};
    sp.timestamp = now_us();
    sp.position = nan3();
    sp.velocity = latest_vel_sp_ned_;
    sp.acceleration = nan3();
    sp.yaw = std::numeric_limits<float>::quiet_NaN();
    sp.yawspeed = std::numeric_limits<float>::quiet_NaN();
    sp_pub_->publish(sp);
  }

  void publish_offboard_heartbeat() {
    px4_msgs::msg::OffboardControlMode msg{};
    msg.timestamp = now_us();
    msg.position = false;
    msg.velocity = true;
    msg.acceleration = false;
    msg.attitude = false;
    msg.body_rate = false;
    offboard_pub_->publish(msg);
  }

  bool enable_depth_camera_{true};
  double publish_period_sec_{0.05};
  double offboard_period_sec_{0.05};
  int nav_state_{-1};
  int depth_frame_count_{0};
  std::array<float, 3> goal_ned_{};
  std::array<float, 3> current_pos_ned_{};
  std::array<float, 3> current_vel_ned_{};
  std::array<float, 3> latest_pos_sp_ned_{};
  std::array<float, 3> latest_vel_sp_ned_{};

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
  rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<px4_msgs::msg::VehicleStatus>::SharedPtr status_sub_;
  rclcpp::Publisher<px4_msgs::msg::TrajectorySetpoint>::SharedPtr sp_pub_;
  rclcpp::Publisher<px4_msgs::msg::OffboardControlMode>::SharedPtr offboard_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr mpc_label_path_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr mpc_traj_pub_;
  rclcpp::TimerBase::SharedPtr publish_timer_;
  rclcpp::TimerBase::SharedPtr offboard_timer_;
  rclcpp::TimerBase::SharedPtr solver_timer_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<MpcOptimalPlannerCpp>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
