#include <models/inference.hpp>

#include <logging/logger.hpp>

#include <opencv2/imgproc.hpp>

#include <array>
#include <chrono>
#include <cstring>
#include <future>
#include <vector>

namespace visionpilot::models {

namespace {

constexpr int NET_W    = AutoDrive::NET_W;
constexpr int NET_H    = AutoDrive::NET_H;
constexpr int CHW_SIZE = AutoDrive::CHW_SIZE;

constexpr float MEAN[3] = {0.485f, 0.456f, 0.406f};
constexpr float STD[3]  = {0.229f, 0.224f, 0.225f};

std::vector<float> chw_imagenet(const cv::Mat& bgr)
{
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    cv::Mat f32;
    rgb.convertTo(f32, CV_32FC3, 1.0 / 255.0);
    std::vector<cv::Mat> ch(3);
    cv::split(f32, ch);
    std::vector<float> out(static_cast<std::size_t>(CHW_SIZE));
    for (int c = 0; c < 3; ++c) {
        float*       dst = out.data() + c * NET_H * NET_W;
        const float* src = reinterpret_cast<const float*>(ch[c].data);
        for (int i = 0; i < NET_H * NET_W; ++i)
            dst[i] = (src[i] - MEAN[c]) / STD[c];
    }
    return out;
}

std::vector<float> chw_01(const cv::Mat& bgr)
{
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    cv::Mat f32;
    rgb.convertTo(f32, CV_32FC3, 1.0 / 255.0);
    std::vector<cv::Mat> ch(3);
    cv::split(f32, ch);
    std::vector<float> out(static_cast<std::size_t>(CHW_SIZE));
    for (int c = 0; c < 3; ++c)
        std::memcpy(out.data() + c * NET_H * NET_W, ch[c].data,
                    static_cast<std::size_t>(NET_H * NET_W) * sizeof(float));
    return out;
}

}  // namespace

void LatencyStats::update(double pre_, double ad_, double as_, double asp_, double wall_)
{
    auto ema = [this](double& e, double v) { e = ok ? e * 0.9 + v * 0.1 : v; };
    ema(pre, pre_);
    ema(ad, ad_);
    ema(as, as_);
    ema(asp, asp_);
    ema(wall, wall_);
    ok = true;
}

void LatencyStats::print() const
{
    if (!ok) return;
    VP_INFO("Latency[EMA]  pre=%.1f  AD=%.1f  AS=%.1f  ASp=%.1f  wall=%.1f ms  (%.0f fps)",
            pre, ad, as, asp, wall, 1000.0 / wall);
}

void LatencyStats::reset() { *this = {}; }

InferencePipeline::InferencePipeline(engine::OnnxEngine& engine, const InferenceConfig& cfg)
    : auto_drive_(engine, cfg.autodrive_model)
    , auto_steer_(engine, cfg.autosteer_model)
    , auto_speed_(engine, cfg.autospeed_model)
{
    fusion::LongitudinalFusion::Config lc;
    lc.homography_path = cfg.homography_path;
    lc.debug           = cfg.fusion_debug;
    long_fusion_ = fusion::LongitudinalFusion{lc};

    fusion::LateralFusion::Config latc;
    latc.homography_path = cfg.homography_path;
    latc.debug           = cfg.fusion_debug;
    lat_fusion_ = fusion::LateralFusion{latc};
}

std::optional<InferenceFrameResult> InferencePipeline::process(const cv::Mat& preprocessed)
{
    using Clock = std::chrono::steady_clock;
    using Ms    = std::chrono::duration<double, std::milli>;

    prev_frame_ = curr_frame_.empty() ? preprocessed.clone() : curr_frame_;
    curr_frame_ = preprocessed.clone();
    if (frame_buf_count_ < 1)
        frame_buf_count_ = 1;
    else
        frame_buf_count_ = 2;

    ++frame_count_;
    if (frame_buf_count_ < 2)
        return std::nullopt;

    auto t0 = Clock::now();
    auto prev_imn = chw_imagenet(prev_frame_);
    auto curr_imn = chw_imagenet(curr_frame_);
    auto curr_01  = chw_01(curr_frame_);
    const double ms_pre = Ms(Clock::now() - t0).count();

    auto t_wall = Clock::now();
    auto f_drive = std::async(std::launch::async, [&] {
        auto t = Clock::now();
        return std::make_pair(auto_drive_.infer(prev_imn.data(), curr_imn.data()),
                              Ms(Clock::now() - t).count());
    });
    auto f_steer = std::async(std::launch::async, [&] {
        auto t = Clock::now();
        return std::make_pair(auto_steer_.infer(curr_01.data()), Ms(Clock::now() - t).count());
    });
    auto f_speed = std::async(std::launch::async, [&] {
        auto t = Clock::now();
        return std::make_pair(auto_speed_.infer(curr_01.data()), Ms(Clock::now() - t).count());
    });

    auto [res_drive, ms_drive] = f_drive.get();
    auto [res_steer, ms_steer] = f_steer.get();
    auto [res_speed, ms_speed] = f_speed.get();
    const double ms_wall = Ms(Clock::now() - t_wall).count();

    InferenceFrameResult out;
    out.frame_id   = frame_count_;
    out.wall_ms  = ms_wall;
    out.pre_ms   = ms_pre;
    out.ad_ms    = ms_drive;
    out.as_ms    = ms_steer;
    out.asp_ms   = ms_speed;
    out.auto_drive = res_drive;
    out.auto_steer = res_steer;
    out.auto_speed = res_speed;
    out.cipo       = long_fusion_.update(res_drive, res_speed, preprocessed);
    out.lateral    = lat_fusion_.update(res_steer, res_drive);

    stats_.update(ms_pre, ms_drive, ms_steer, ms_speed, ms_wall);
    return out;
}

void InferencePipeline::reset()
{
    prev_frame_.release();
    curr_frame_.release();
    frame_buf_count_ = 0;
    frame_count_ = 0;
    stats_.reset();
    long_fusion_.reset();
    lat_fusion_.reset();
}

}  // namespace visionpilot::models
