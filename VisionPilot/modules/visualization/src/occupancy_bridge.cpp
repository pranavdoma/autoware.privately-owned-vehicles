#include <visualization/occupancy_bridge.hpp>
#include <visualization/visual_interface.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace visualization
{
namespace occupancy
{

namespace
{

constexpr int kNetW = 1024;
constexpr int kNetH = 512;

}  // namespace

Scene make_scene(
  const visionpilot::models::InferenceFrameResult & r, const Plan & /*plan*/,
  const cv::Mat & H_resized, float ego_speed_ms)
{
  Scene s;

  if (!H_resized.empty()) {
    cv::Mat H64;
    H_resized.convertTo(H64, CV_64F);
    H64.convertTo(s.H_px2world, CV_32F);
  }

  if (r.lateral.path_valid) {
    s.path_a = r.lateral.path_a;
    s.path_b = r.lateral.path_b;
    s.path_c = r.lateral.path_c;
    s.path_valid = true;
  }
  // Always take filtered CTE/yaw so ego can animate during lane changes
  // even when the RANSAC path briefly drops out.
  if (r.lateral.valid) {
    s.cte_m = r.lateral.cte_m;
    s.yaw_rad = r.lateral.yaw_rad;
  } else if (r.lateral.path_valid) {
    s.cte_m = r.lateral.raw_cte_m;
    s.yaw_rad = r.lateral.raw_yaw_rad;
  }

  if (r.auto_steer.valid && !s.H_px2world.empty()) {
    constexpr int kSteerN = 64;
    s.lane_world.reserve(kSteerN);
    std::vector<cv::Point2f> src;
    src.reserve(kSteerN);
    for (int i = 0; i < kSteerN; ++i) {
      if (r.auto_steer.h_vector[static_cast<size_t>(i)] < 0.5f) continue;
      const float u = r.auto_steer.xp[static_cast<size_t>(i)] * static_cast<float>(kNetW);
      const float v =
        static_cast<float>(i) * (static_cast<float>(kNetH - 1) / static_cast<float>(kSteerN - 1));
      src.emplace_back(u, v);
    }
    if (!src.empty()) {
      std::vector<cv::Point2f> dst;
      cv::perspectiveTransform(src, dst, s.H_px2world);
      for (const auto & p : dst) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y)) continue;
        if (p.x < -1.f || p.x > 80.f) continue;
        s.lane_world.push_back(p);
      }
    }
  }

  s.detections.reserve(r.auto_speed.detections.size());
  for (const auto & d : r.auto_speed.detections) {
    s.detections.push_back({d.x1, d.y1, d.x2, d.y2, d.score, d.class_id});
  }

  s.cipo_valid = r.cipo.valid;
  s.cipo_distance_m = r.cipo.distance_m;
  s.ego_speed_ms = ego_speed_ms;

  const auto & radar = r.cipo.radar;
  s.radar_enabled = radar.enabled;
  if (radar.enabled && !radar.points.empty()) {
    const float ego = (ego_speed_ms > 0.5f) ? ego_speed_ms : 0.f;
    const bool have_ego = ego > 0.5f;
    s.radar_points.resize(radar.points.size());
    for (size_t i = 0; i < radar.points.size(); ++i) {
      const auto & p = radar.points[i];
      Scene::RadarReturn rr;
      rr.x = p.range_m * std::cos(p.azimuth_rad);
      rr.y = p.range_m * std::sin(p.azimuth_rad);
      rr.range_rate = p.range_rate;
      rr.in_match = (i < radar.in_match.size() && radar.in_match[i] != 0);
      const float abs_v = have_ego
                            ? std::abs(p.range_rate + ego * std::cos(p.azimuth_rad))
                            : std::abs(p.range_rate);
      rr.moving = abs_v > 1.0f;
      rr.cluster_id = -1;
      s.radar_points[i] = rr;
    }

    auto moving_of = [&](float range_rate, float azimuth_rad) {
      const float abs_v = have_ego ? std::abs(range_rate + ego * std::cos(azimuth_rad))
                                   : std::abs(range_rate);
      return abs_v > 1.0f;
    };

    s.radar_clusters.reserve(radar.clusters.size() + 1);
    for (int ci = 0; ci < static_cast<int>(radar.clusters.size()); ++ci) {
      const auto & c = radar.clusters[static_cast<size_t>(ci)];
      float xmin = 1e9f, xmax = -1e9f, ymin = 1e9f, ymax = -1e9f;
      int n_ok = 0;
      int n_match = 0;
      for (int idx : c.members) {
        if (idx < 0 || idx >= static_cast<int>(s.radar_points.size())) continue;
        auto & pt = s.radar_points[static_cast<size_t>(idx)];
        pt.cluster_id = ci;
        xmin = std::min(xmin, pt.x);
        xmax = std::max(xmax, pt.x);
        ymin = std::min(ymin, pt.y);
        ymax = std::max(ymax, pt.y);
        ++n_ok;
        if (pt.in_match) ++n_match;
      }
      if (n_ok == 0) continue;
      constexpr float kPad = 0.35f;
      constexpr float kMin = 0.80f;
      Scene::RadarCluster box;
      box.x0 = xmin - kPad;
      box.x1 = xmax + kPad;
      box.y0 = ymin - kPad;
      box.y1 = ymax + kPad;
      if (box.x1 - box.x0 < kMin) {
        const float mid = 0.5f * (box.x0 + box.x1);
        box.x0 = mid - 0.5f * kMin;
        box.x1 = mid + 0.5f * kMin;
      }
      if (box.y1 - box.y0 < kMin) {
        const float mid = 0.5f * (box.y0 + box.y1);
        box.y0 = mid - 0.5f * kMin;
        box.y1 = mid + 0.5f * kMin;
      }
      box.cx = c.range_m * std::cos(c.azimuth_rad);
      box.cy = c.range_m * std::sin(c.azimuth_rad);
      box.id = ci;
      box.n = n_ok;
      box.is_match = (n_match * 2 >= n_ok);
      box.moving = moving_of(c.range_rate, c.azimuth_rad);
      s.radar_clusters.push_back(box);
    }

    // FOV-window association can pick a set that is not one DBSCAN cluster.
    // Keep a dedicated match volume so Occupancy still shows the CIPO set.
    bool match_covered = false;
    for (const auto & box : s.radar_clusters)
      if (box.is_match) match_covered = true;
    if (!match_covered) {
      float xmin = 1e9f, xmax = -1e9f, ymin = 1e9f, ymax = -1e9f;
      int n_ok = 0;
      for (const auto & pt : s.radar_points) {
        if (!pt.in_match) continue;
        xmin = std::min(xmin, pt.x);
        xmax = std::max(xmax, pt.x);
        ymin = std::min(ymin, pt.y);
        ymax = std::max(ymax, pt.y);
        ++n_ok;
      }
      if (n_ok > 0) {
        constexpr float kPad = 0.35f;
        Scene::RadarCluster box;
        box.x0 = xmin - kPad;
        box.x1 = xmax + kPad;
        box.y0 = ymin - kPad;
        box.y1 = ymax + kPad;
        box.cx = 0.5f * (box.x0 + box.x1);
        box.cy = 0.5f * (box.y0 + box.y1);
        box.id = static_cast<int>(s.radar_clusters.size());
        box.n = n_ok;
        box.is_match = true;
        box.moving = false;
        s.radar_clusters.push_back(box);
      }
    }
  }

  // AutoDrive-only CIPO: AD confirms in-path object, AutoSpeed has no bbox.
  static constexpr float kDMaxM = 150.f;
  if (r.auto_drive.valid && r.auto_drive.flag_prob >= 0.40f && !r.cipo.cipo_raw_found) {
    s.ad_cipo_only = true;
    s.ad_distance_m = kDMaxM * (1.f - r.auto_drive.dist_normalized);
  }

  return s;
}

cv::Mat build_frame(
  const visionpilot::models::InferenceFrameResult & result, const Plan & plan,
  const cv::Mat & H_resized, float ego_speed_ms)
{
  return render(make_scene(result, plan, H_resized, ego_speed_ms));
}

void publish(
  VisualInterface * visual_interface, const visionpilot::models::InferenceFrameResult & result,
  const Plan & plan, const cv::Mat & H_resized, float ego_speed_ms)
{
  if (!visual_interface) return;
  visual_interface->set_aux_frame(build_frame(result, plan, H_resized, ego_speed_ms));
}

}  // namespace occupancy
}  // namespace visualization
