#pragma once

#include "models/model_types.hpp"
#include "models/onnx_session.hpp"

#include <onnxruntime_cxx_api.h>

#include <memory>
#include <string>
#include <vector>

namespace visionpilot::models {

// ─── Interface ────────────────────────────────────────────────────────────────
// Abstract base for the AutoDrive two-frame unified model.
//
// Preprocessing contract (caller's responsibility before calling infer()):
//   • Resize both frames to NET_W × NET_H (1024 × 512)
//   • Convert BGR → RGB
//   • Normalise with ImageNet stats (mean=[0.485,0.456,0.406],
//                                    std =[0.229,0.224,0.225])
//   • Layout: CHW float32, size = 3 × NET_H × NET_W = CHW_SIZE elements
//
// Output domain conversion (caller's responsibility after infer()):
//   distance_m    = D_MAX_M   * (1.0f - out.dist_normalized)   // D_MAX_M = 150.0
//   curvature_1pm = CURV_SCALE * out.curvature_raw
class AutoDriveBase {
public:
    virtual ~AutoDriveBase() = default;

    // prev_chw / curr_chw : float32 CHW buffers, CHW_SIZE elements each.
    virtual AutoDriveOutput infer(const float* prev_chw, const float* curr_chw) = 0;
};

// ─── ONNX Runtime implementation ─────────────────────────────────────────────
class AutoDriveOnnx final : public AutoDriveBase {
public:
    static constexpr int NET_H    = 512;
    static constexpr int NET_W    = 1024;
    static constexpr int CHW_SIZE = 3 * NET_H * NET_W;  // 1 572 864

    explicit AutoDriveOnnx(const OnnxSessionConfig& cfg);
    ~AutoDriveOnnx() override = default;

    AutoDriveOutput infer(const float* prev_chw, const float* curr_chw) override;

private:
    Ort::Env                       env_;
    std::unique_ptr<Ort::Session>  session_;
    Ort::MemoryInfo                mem_info_;

    std::vector<std::string>  in_name_strs_;
    std::vector<const char*>  in_names_;

    std::vector<std::string>  out_name_strs_;
    std::vector<const char*>  out_names_;

    std::vector<int64_t>  frame_shape_;  // {1, 3, NET_H, NET_W}
};

}  // namespace visionpilot::models
