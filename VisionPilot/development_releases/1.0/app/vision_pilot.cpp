#include "vision_pilot_config.hpp"

#include <camera_subscriber/ros2_to_opencv.hpp>
#include <v4l2_interface/v4l2_reader.hpp>
#include <visualization/visualization.hpp>

#include <engine/onnx_engine.hpp>
#include <models/auto_drive.hpp>
#include <models/auto_steer.hpp>
#include <models/auto_speed.hpp>

#include <opencv2/opencv.hpp>

#include <array>
#include <chrono>
#include <cstring>
#include <future>
#include <iostream>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace {

// First argv token that is not a config flag (expected: capture mode 0 or 1).
int index_of_mode_arg(int argc, char** argv)
{
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--config" || arg == "-c") {
            ++i;
            continue;
        }
        if (arg.rfind("--config=", 0) == 0) {
            continue;
        }
        return i;
    }
    return -1;
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Per-frame result bundle
// ─────────────────────────────────────────────────────────────────────────────

struct FrameOutputs {
    models::AutoDriveOutput auto_drive;
    models::AutoSteerOutput auto_steer;
    models::AutoSpeedOutput auto_speed;
    uint64_t frame_id = 0;
};

// ─────────────────────────────────────────────────────────────────────────────
// InferencePipeline
//
// Owns the 2-slot circular frame buffer and the three model instances.
// Caller is responsible for spatial pre-processing (warp/crop/resize to
// NET_W × NET_H) before calling process().  Normalization to CHW float32
// is done here, once per normalization variant, before any threads start.
//
// Parallelism:
//   All three models run concurrently via std::async.
//   process() blocks until the slowest model finishes:
//     wall-time ≈ max(T_autodrive, T_autosteer, T_autospeed)
//
// Thread safety:
//   process() is NOT thread-safe; call from a single capture/inference thread.
// ─────────────────────────────────────────────────────────────────────────────

class InferencePipeline {
public:
    static constexpr int NET_H    = 512;
    static constexpr int NET_W    = 1024;
    static constexpr int CHW_SIZE = 3 * NET_H * NET_W;  // 1 572 864 floats

    InferencePipeline(engine::OnnxEngine& ort_engine, const VisionPilotConfig& cfg)
        : auto_drive_(ort_engine, cfg.autodrive_model)
        , auto_steer_(ort_engine, cfg.autosteer_model)
        , auto_speed_(ort_engine, cfg.autospeed_model)
    {}

    // Push a pre-processed BGR frame (must already be NET_W × NET_H).
    // Returns nullopt on the very first call — AutoDrive needs prev + curr.
    std::optional<FrameOutputs> process(const cv::Mat& frame_bgr)
    {
        // ── 1. Write into the circular buffer ────────────────────────────
        // Slots alternate: slot 0 ↔ slot 1 every frame.
        // After the write:
        //   frame_buffer_[write_idx_]          = current frame
        //   frame_buffer_[(write_idx_+1) % 2]  = previous frame
        write_idx_ = (write_idx_ + 1) % 2;
        frame_buffer_[write_idx_] = frame_bgr.clone();
        ++frame_count_;

        if (frame_count_ < 2) {
            // First frame stored; no previous frame yet — skip inference.
            return std::nullopt;
        }

        const int prev_idx = (write_idx_ + 1) % 2;

        // ── 2. Build CHW tensors before launching threads ─────────────────
        //
        //  AutoDrive  needs ImageNet-normalised CHW for both prev and curr.
        //  AutoSteer  needs [0,1]-scaled CHW for curr only.
        //  AutoSpeed  needs [0,1]-scaled CHW for curr only (shared with steer).
        //
        //  These vectors are heap-allocated here and captured by reference
        //  inside the lambdas below.  All futures are .get()'d before this
        //  function returns, so the lifetimes are safe.
        std::vector<float> prev_imagenet = to_chw_imagenet(frame_buffer_[prev_idx]);
        std::vector<float> curr_imagenet = to_chw_imagenet(frame_buffer_[write_idx_]);
        std::vector<float> curr_01       = to_chw_01(frame_buffer_[write_idx_]);

        // ── 3. Launch all three inferences concurrently ──────────────────
        auto f_drive = std::async(std::launch::async, [&] {
            return auto_drive_.infer(prev_imagenet.data(), curr_imagenet.data());
        });

        auto f_steer = std::async(std::launch::async, [&] {
            return auto_steer_.infer(curr_01.data());
        });

        // AutoSpeed shares curr_01 — read-only, no race condition.
        auto f_speed = std::async(std::launch::async, [&] {
            return auto_speed_.infer(curr_01.data());
        });

        // ── 4. Barrier — block until every model finishes ────────────────
        //
        //  .get() calls are sequential but the underlying work ran in
        //  parallel.  Total wall time ≈ max(T_drive, T_steer, T_speed).
        //
        //  If any model throws, the exception propagates here, leaving the
        //  remaining futures to be collected before they go out of scope.
        FrameOutputs out;
        out.auto_drive = f_drive.get();
        out.auto_steer = f_steer.get();
        out.auto_speed = f_speed.get();
        out.frame_id   = frame_count_;

        return out;
    }

private:
    // ── Normalization helpers ─────────────────────────────────────────────

    static constexpr float MEAN[3] = {0.485f, 0.456f, 0.406f};
    static constexpr float STD[3]  = {0.229f, 0.224f, 0.225f};

    // BGR → RGB → float32 → ImageNet normalise → CHW
    static std::vector<float> to_chw_imagenet(const cv::Mat& bgr)
    {
        cv::Mat rgb;
        cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

        cv::Mat f32;
        rgb.convertTo(f32, CV_32FC3, 1.0 / 255.0);

        std::vector<cv::Mat> ch(3);
        cv::split(f32, ch);

        std::vector<float> out(CHW_SIZE);
        for (int c = 0; c < 3; ++c) {
            float*       dst = out.data() + c * NET_H * NET_W;
            const float* src = reinterpret_cast<const float*>(ch[c].data);
            for (int i = 0; i < NET_H * NET_W; ++i)
                dst[i] = (src[i] - MEAN[c]) / STD[c];
        }
        return out;
    }

    // BGR → RGB → float32 [0, 1] → CHW
    static std::vector<float> to_chw_01(const cv::Mat& bgr)
    {
        cv::Mat rgb;
        cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

        cv::Mat f32;
        rgb.convertTo(f32, CV_32FC3, 1.0 / 255.0);

        std::vector<cv::Mat> ch(3);
        cv::split(f32, ch);

        std::vector<float> out(CHW_SIZE);
        for (int c = 0; c < 3; ++c) {
            float* dst = out.data() + c * NET_H * NET_W;
            std::memcpy(dst, ch[c].data, NET_H * NET_W * sizeof(float));
        }
        return out;
    }

    // ── State ─────────────────────────────────────────────────────────────

    models::AutoDrive auto_drive_;
    models::AutoSteer auto_steer_;
    models::AutoSpeed auto_speed_;

    // 2-slot circular buffer of pre-processed BGR frames.
    // write_idx_ starts at 1 so the first write goes to slot 0.
    std::array<cv::Mat, 2> frame_buffer_;
    int      write_idx_   = 1;
    uint64_t frame_count_ = 0;
};

// ─────────────────────────────────────────────────────────────────────────────
// Spatial pre-processing  (warp / crop / resize → NET_W × NET_H)
//
// In production this will apply the homography warp + 50° FOV crop.
// For now: simple resize as a placeholder.
// ─────────────────────────────────────────────────────────────────────────────

static cv::Mat spatial_preprocess(const cv::Mat& raw)
{
    cv::Mat out;
    cv::resize(raw, out,
               cv::Size(InferencePipeline::NET_W, InferencePipeline::NET_H),
               0, 0, cv::INTER_LINEAR);
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
    const int mode_idx = index_of_mode_arg(argc, argv);
    if (mode_idx < 0 || mode_idx >= argc) {
        std::cout << "Usage: " << argv[0] << " [--config path] <mode> [args...]\n"
                  << "  Config (first match wins):\n"
                  << "    --config / -c <path>  |  $VISIONPILOT_CONFIG\n"
                  << "    ./config/vision_pilot.conf  |  ./vision_pilot.conf\n"
                  << "  Copy and edit: config/vision_pilot.conf.example\n"
                  << "  mode 0 (ROS2): [topic]           default: /camera/image\n"
                  << "  mode 1 (V4L2): [device] [fps]   default: /dev/video0  10\n";
        return 1;
    }

    const std::string config_path = resolve_vision_pilot_config_path(argc, argv);
    if (config_path.empty()) {
        std::cerr << "[VisionPilot] No config file found.\n"
                  << "  cp config/vision_pilot.conf.example config/vision_pilot.conf\n"
                  << "  Or: export VISIONPILOT_CONFIG=/path/to/vision_pilot.conf\n";
        return 1;
    }

    VisionPilotConfig cfg;
    try {
        cfg = load_vision_pilot_config(config_path);
    } catch (const std::exception& e) {
        std::cerr << "[VisionPilot] Config error: " << e.what() << "\n";
        return 1;
    }

    std::cout << "[VisionPilot] Config: " << config_path << "\n"
              << "  autodrive: " << cfg.autodrive_model << "\n"
              << "  autosteer: " << cfg.autosteer_model << "\n"
              << "  autospeed: " << cfg.autospeed_model << "\n"
              << "  provider:  " << cfg.engine.provider << "\n";

    engine::OnnxEngine ort_engine(cfg.engine);
    InferencePipeline  pipeline(ort_engine, cfg);

    const int mode = std::stoi(argv[mode_idx]);

    // ════════════════════════════════════════════════════════════════════════
    // ROS2 MODE
    // ════════════════════════════════════════════════════════════════════════
    if (mode == 0) {

        const std::string topic = (mode_idx + 1 < argc) ? argv[mode_idx + 1] : "/camera/image";
        std::cout << "[VisionPilot] ROS2 mode | topic: " << topic << "\n";

        camera_subscriber::ROS2ImageSubscriber ros2_sub(topic);

        while (true) {
            auto [has_frame, frame] = ros2_sub.get_latest_frame();

            if (!has_frame || frame.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }

            cv::Mat preprocessed = spatial_preprocess(frame);
            auto result = pipeline.process(preprocessed);

            if (!result) {
                // First frame — buffer warming up, nothing to output yet.
                continue;
            }

            // ── Downstream: use result->auto_drive / auto_steer / auto_speed
            // TODO: hand off to planning / control / visualization modules.

            auto stats = ros2_sub.get_stats();
            std::vector<std::string> overlay = {
                "topic: "            + topic,
                "frame: "            + std::to_string(result->frame_id),
                "frames received: "  + std::to_string(stats.frames_received),
                "frames dropped: "   + std::to_string(stats.frames_dropped),
            };
            visualization::render_frame(frame, "VisionPilot", overlay);
        }

        visualization::close_windows();

    // ════════════════════════════════════════════════════════════════════════
    // V4L2 MODE
    // ════════════════════════════════════════════════════════════════════════
    } else if (mode == 1) {

        const std::string device_path = (mode_idx + 1 < argc) ? argv[mode_idx + 1] : "/dev/video0";
        const uint32_t    target_fps  = (mode_idx + 2 < argc)
            ? static_cast<uint32_t>(std::stoi(argv[mode_idx + 2])) : 10;

        std::cout << "[VisionPilot] V4L2 mode | device: " << device_path
                  << "  fps: " << target_fps << "\n";

        v4l2_interface::V4L2Reader v4l2_reader(device_path, target_fps);
        if (!v4l2_reader.is_device_open()) {
            std::cerr << "[VisionPilot] Failed to open V4L2 device: " << device_path << "\n";
            return 1;
        }

        while (true) {
            auto [has_frame, frame] = v4l2_reader.get_latest_frame();

            if (!has_frame || frame.empty()) continue;

            cv::Mat preprocessed = spatial_preprocess(frame);
            auto result = pipeline.process(preprocessed);

            if (!result) continue;

            // ── Downstream: use result->auto_drive / auto_steer / auto_speed
            // TODO: hand off to planning / control / visualization modules.

            auto stats = v4l2_reader.get_stats();
            std::vector<std::string> overlay = {
                "device: "    + device_path,
                "frame: "     + std::to_string(result->frame_id),
                "fps: "       + std::to_string(static_cast<int>(stats.current_fps)),
                "resolution: "+ std::to_string(stats.current_width) + "x"
                              + std::to_string(stats.current_height),
            };
            visualization::render_frame(frame, "VisionPilot", overlay);
        }

        visualization::close_windows();

    } else {
        std::cerr << "[VisionPilot] Invalid mode. Use 0 (ROS2) or 1 (V4L2).\n";
        return 1;
    }

    return 0;
}
