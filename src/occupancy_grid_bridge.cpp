#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <foxglove_msgs/msg/line_primitive.hpp>
#include <foxglove_msgs/msg/packed_element_field.hpp>
#include <foxglove_msgs/msg/scene_update.hpp>
#include <foxglove_msgs/msg/voxel_grid.hpp>
#include <hippo_guidance_msgs/msg/occupancy_grid3_d.hpp>
#include <rclcpp/rclcpp.hpp>
#include <yaml-cpp/yaml.h>

namespace {

using Grid = hippo_guidance_msgs::msg::OccupancyGrid3D;
using VoxelGrid = foxglove_msgs::msg::VoxelGrid;
using SceneUpdate = foxglove_msgs::msg::SceneUpdate;

constexpr size_t kMaxVoxels = 50'000'000;
constexpr uint32_t kCellStride = 5;
constexpr std::array<const char*, 4> kStateNames{
    "unknown", "free", "occupied", "buffer"};

struct Color {
  std::array<uint8_t, 3> rgb{};
  uint8_t alpha{};
};

uint8_t ByteValue(const YAML::Node& value) {
  const int number = value.as<int>();
  if (number < 0 || number > 255) {
    throw std::runtime_error("voxel color value must be between 0 and 255");
  }
  return static_cast<uint8_t>(number);
}

std::array<Color, 4> LoadColors(const std::string& path) {
  const auto config = YAML::LoadFile(path);
  std::array<Color, 4> colors;
  for (size_t state = 0; state < colors.size(); ++state) {
    const auto entry = config[kStateNames[state]];
    const auto rgb = entry["rgb"];
    if (!rgb.IsSequence() || rgb.size() != 3) {
      throw std::runtime_error(std::string("invalid rgb for ") + kStateNames[state]);
    }
    for (size_t channel = 0; channel < 3; ++channel) {
      colors[state].rgb[channel] = ByteValue(rgb[channel]);
    }
    colors[state].alpha = ByteValue(entry["foxglove_alpha"]);
  }
  return colors;
}

struct Dimensions {
  uint32_t x;
  uint32_t y;
  size_t count;
};

Dimensions Validate(const Grid& map) {
  const auto& lower = map.position_min;
  const auto& upper = map.position_max;
  const std::array<double, 3> sizes{
      upper.x - lower.x, upper.y - lower.y, upper.z - lower.z};
  if (map.resolution_per_meter == 0 || map.header.frame_id.empty()) {
    throw std::runtime_error("grid needs a frame and positive resolution");
  }
  size_t count = 1;
  std::array<uint32_t, 3> dims{};
  for (size_t axis = 0; axis < 3; ++axis) {
    const double dimension = std::ceil(sizes[axis] * map.resolution_per_meter);
    if (!std::isfinite(sizes[axis]) || sizes[axis] <= 0.0 ||
        !std::isfinite(dimension) || dimension < 1.0 ||
        dimension > static_cast<double>(std::numeric_limits<uint32_t>::max())) {
      throw std::runtime_error("invalid grid bounds");
    }
    dims[axis] = static_cast<uint32_t>(dimension);
    if (count > kMaxVoxels / dims[axis]) {
      throw std::runtime_error("grid exceeds 50 million voxels");
    }
    count *= dims[axis];
  }
  if (map.data.size() != count) {
    throw std::runtime_error("voxel data has the wrong length");
  }
  if (count * kCellStride > std::numeric_limits<uint32_t>::max() ||
      static_cast<uint64_t>(dims[0]) * dims[1] * kCellStride >
          std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("grid strides exceed uint32 range");
  }
  return {dims[0], dims[1], count};
}

VoxelGrid MakeGrid(const Grid& map, const Dimensions& dims) {
  VoxelGrid grid;
  grid.frame_id = map.header.frame_id;
  grid.pose.position = map.position_min;
  grid.pose.orientation.w = 1.0;
  grid.column_count = dims.x;
  grid.row_count = dims.y;
  const double cell_size = 1.0 / map.resolution_per_meter;
  grid.cell_size.x = grid.cell_size.y = grid.cell_size.z = cell_size;
  grid.cell_stride = kCellStride;
  grid.row_stride = dims.x * kCellStride;
  grid.slice_stride = dims.y * grid.row_stride;
  for (uint32_t offset = 0; offset < kCellStride; ++offset) {
    foxglove_msgs::msg::PackedElementField field;
    field.name = std::array<const char*, kCellStride>{
        "state", "red", "green", "blue", "alpha"}[offset];
    field.offset = offset;
    field.type = foxglove_msgs::msg::PackedElementField::UINT8;
    grid.fields.push_back(std::move(field));
  }
  grid.data.resize(dims.count * kCellStride);
  return grid;
}

SceneUpdate MakeBounds(const Grid& map) {
  SceneUpdate scene;
  auto& entity = scene.entities.emplace_back();
  entity.frame_id = map.header.frame_id;
  entity.id = "position_bounds";
  auto& line = entity.lines.emplace_back();
  line.type = foxglove_msgs::msg::LinePrimitive::LINE_LIST;
  line.pose.orientation.w = 1.0;
  line.thickness = 2.0;
  line.scale_invariant = true;
  line.color.r = 0.0;
  line.color.g = 0.75;
  line.color.b = 0.85;
  line.color.a = 1.0;
  const std::array<double, 3> lower{
      map.position_min.x, map.position_min.y, map.position_min.z};
  const std::array<double, 3> upper{
      map.position_max.x, map.position_max.y, map.position_max.z};
  line.points.reserve(24);
  for (size_t axis = 0; axis < 3; ++axis) {
    const size_t other_a = (axis + 1) % 3;
    const size_t other_b = (axis + 2) % 3;
    for (int a = 0; a < 2; ++a) {
      for (int b = 0; b < 2; ++b) {
        std::array<double, 3> start = lower;
        start[other_a] = a ? upper[other_a] : lower[other_a];
        start[other_b] = b ? upper[other_b] : lower[other_b];
        auto end = start;
        end[axis] = upper[axis];
        geometry_msgs::msg::Point first, second;
        first.x = start[0]; first.y = start[1]; first.z = start[2];
        second.x = end[0]; second.y = end[1]; second.z = end[2];
        line.points.push_back(first);
        line.points.push_back(second);
      }
    }
  }
  return scene;
}

class OccupancyGridBridge : public rclcpp::Node {
 public:
  OccupancyGridBridge() : Node("occupancy_grid_bridge") {
    const auto color_path = declare_parameter<std::string>(
        "colors_file", ament_index_cpp::get_package_share_directory("hippo_maps") +
                           "/config/voxel_colors.yaml");
    colors_ = LoadColors(color_path);
    auto qos = rclcpp::QoS(1).reliable().transient_local();
    occupied_pub_ = create_publisher<VoxelGrid>(
        declare_parameter<std::string>("output_topic", "/pool/voxel_grid"), qos);
    buffer_pub_ = create_publisher<VoxelGrid>(
        declare_parameter<std::string>("buffer_topic", "/pool/buffer_voxel_grid"), qos);
    bounds_pub_ = create_publisher<SceneUpdate>(
        declare_parameter<std::string>("bounds_topic", "/pool/position_bounds"), qos);
    map_sub_ = create_subscription<Grid>(
        declare_parameter<std::string>("input_topic", "/occupancy_grid_3d"), qos,
        [this](Grid::ConstSharedPtr map) { OnMap(*map); });
  }

 private:
  void OnMap(const Grid& map) {
    try {
      const auto dims = Validate(map);
      auto occupied = MakeGrid(map, dims);
      auto buffer = MakeGrid(map, dims);
      for (size_t i = 0; i < dims.count; ++i) {
        const uint8_t state = map.data[i];
        if (state >= colors_.size()) {
          throw std::runtime_error("grid contains reserved voxel states");
        }
        const auto& color = colors_[state];
        const size_t offset = i * kCellStride;
        auto write = [&](std::vector<uint8_t>& data, bool visible) {
          data[offset] = state;
          data[offset + 1] = color.rgb[0];
          data[offset + 2] = color.rgb[1];
          data[offset + 3] = color.rgb[2];
          data[offset + 4] = visible ? color.alpha : 0;
        };
        write(occupied.data, state == Grid::OCCUPIED);
        write(buffer.data, state == Grid::BUFFER);
      }
      auto bounds = MakeBounds(map);
      const builtin_interfaces::msg::Time stamp = get_clock()->now();
      occupied.timestamp = buffer.timestamp = stamp;
      bounds.entities[0].timestamp = stamp;
      occupied_pub_->publish(occupied);
      buffer_pub_->publish(buffer);
      bounds_pub_->publish(bounds);
      RCLCPP_INFO(get_logger(), "Published %zu voxels per display layer", dims.count);
    } catch (const std::exception& error) {
      RCLCPP_ERROR(get_logger(), "Invalid occupancy grid: %s", error.what());
    }
  }

  std::array<Color, 4> colors_;
  rclcpp::Publisher<VoxelGrid>::SharedPtr occupied_pub_, buffer_pub_;
  rclcpp::Publisher<SceneUpdate>::SharedPtr bounds_pub_;
  rclcpp::Subscription<Grid>::SharedPtr map_sub_;
};

}  // namespace

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<OccupancyGridBridge>());
  } catch (const std::exception& error) {
    RCLCPP_FATAL(rclcpp::get_logger("occupancy_grid_bridge"), "%s", error.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
