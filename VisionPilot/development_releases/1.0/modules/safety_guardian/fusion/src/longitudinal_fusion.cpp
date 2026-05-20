#include <fusion/longitudinal_fusion.hpp>
#include <tracking/object_finder.hpp>    // private — not in public header
#include <logging/logger.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace visionpilot::fusion {

// ─── Construction / destruction ───────────────────────────────────────────────

LongitudinalFusion::LongitudinalFusion()
    : LongitudinalFusion(Config{})
{}

LongitudinalFusion::LongitudinalFusion(Config cfg)
    : cfg_(cfg)
    , rng_(std::random_device{}())
{
    if (cfg_.n_particles < 10)
        throw std::invalid_argument("LongitudinalFusion: n_particles must be >= 10");
    particles_.reserve(static_cast<std::size_t>(cfg_.n_particles));
}

// Destructor + move ops defined here so unique_ptr<ObjectFinder> has a complete type.
LongitudinalFusion::~LongitudinalFusion()                                   = default;
LongitudinalFusion::LongitudinalFusion(LongitudinalFusion&&) noexcept       = default;
LongitudinalFusion& LongitudinalFusion::operator=(LongitudinalFusion&&) noexcept = default;

// ─── Public API ───────────────────────────────────────────────────────────────

void LongitudinalFusion::reset()
{
    particles_.clear();
    initialised_ = false;
    tracker_.reset();
}

CIPOFusionEstimate LongitudinalFusion::update(
    const models::AutoDriveOutput& autodrive,
    const models::AutoSpeedOutput& autospeed,
    const cv::Mat& preprocessed_frame,
    float dt_s)
{
    // ── Step 1: Lazy-init ObjectFinder ────────────────────────────────────────
    // The preprocessed_frame is the 1024×512 center-cropped image fed to the
    // models — the same coordinate space the homography was calibrated in.
    if (!tracker_ && !cfg_.homography_path.empty() && !preprocessed_frame.empty()) {
        try {
            tracker_ = std::make_unique<tracking::ObjectFinder>(
                cfg_.homography_path,
                preprocessed_frame.cols,
                preprocessed_frame.rows);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[LongitudinalFusion] ObjectFinder init failed: %s\n", e.what());
            cfg_.homography_path.clear();
        }
    }

    // ── Step 2: Run ObjectFinder ──────────────────────────────────────────────
    // AutoSpeed bboxes are in 1024×512 model space — exactly the same space as
    // the homography calibration.  No coordinate scaling needed.
    CIPOFusionEstimate est;
    Meas tracker_meas;
    if (tracker_ && autospeed.valid && !preprocessed_frame.empty()) {
        const auto tr = tracker_->update_and_get_cipo(
            autospeed.detections, preprocessed_frame);
        if (tr.cipo.exists) {
            est.tracker_found   = true;
            est.tracker_id      = tr.cipo.track_id;
            est.tracker_dist_m  = tr.cipo.distance_m;
            est.tracker_vel_ms  = tr.cipo.velocity_ms;
            est.cut_in_detected = tr.cut_in_detected;

            tracker_meas.distance_m = tr.cipo.distance_m;
            tracker_meas.stddev_m   = cfg_.tracker_noise_m;
            tracker_meas.valid      = true;
        }
    }

    // ── Step 3: AutoDrive distance measurement ────────────────────────────────
    static constexpr float D_MAX = 150.f;
    Meas autodrive_meas;
    if (autodrive.valid) {
        autodrive_meas.distance_m = D_MAX * (1.f - autodrive.dist_normalized);
        autodrive_meas.stddev_m   = cfg_.autodrive_noise_m;
        autodrive_meas.valid      = true;
    }

    // ── Step 4: Particle filter ───────────────────────────────────────────────
    const float dt = (dt_s > 1e-6f) ? dt_s : cfg_.dt_s;

    if (!initialised_) {
        if (!autodrive_meas.valid) return est;  // tracker fields already set above
        init_from(autodrive_meas.distance_m, autodrive_meas.stddev_m);
        initialised_ = true;
    } else {
        predict(dt);
    }

    // Log-weights are accumulated in weight_update; no normalise() needed here.
    // linear_weights() applies log-sum-exp on-the-fly for numerical stability.
    weight_update(autodrive_meas, tracker_meas);
    if (effective_n() < 0.5f * static_cast<float>(cfg_.n_particles)) resample();

    // ── Step 5: Posterior statistics (weighted mean + variance) ──────────────
    const auto w = linear_weights();
    const std::size_t N = particles_.size();

    float mean_d = 0.f, mean_v = 0.f;
    for (std::size_t i = 0; i < N; ++i) {
        mean_d += w[i] * particles_[i].distance_m;
        mean_v += w[i] * particles_[i].velocity_ms;
    }

    float var_d = 0.f, var_v = 0.f;
    for (std::size_t i = 0; i < N; ++i) {
        const float dd = particles_[i].distance_m  - mean_d;
        const float dv = particles_[i].velocity_ms - mean_v;
        var_d += w[i] * dd * dd;
        var_v += w[i] * dv * dv;
    }

    est.valid              = true;
    est.distance_m         = mean_d;
    est.velocity_ms        = mean_v;
    est.distance_stddev_m  = std::sqrt(std::max(0.f, var_d));
    est.velocity_stddev_ms = std::sqrt(std::max(0.f, var_v));

    // ── Step 6: Debug log — compare all three estimates side-by-side ─────────
    if (cfg_.debug) {
        char ad_buf[32], tr_buf[64];
        if (autodrive_meas.valid)
            std::snprintf(ad_buf, sizeof(ad_buf), "%.1f m", autodrive_meas.distance_m);
        else
            std::snprintf(ad_buf, sizeof(ad_buf), "(invalid)");

        if (tracker_meas.valid)
            std::snprintf(tr_buf, sizeof(tr_buf), "%.1f m  v=%.2f m/s  (id=%d)",
                          est.tracker_dist_m, est.tracker_vel_ms, est.tracker_id);
        else
            std::snprintf(tr_buf, sizeof(tr_buf), "(no CIPO)");

        VP_INFO("[Fusion] AutoDrive=%s | Tracker=%s | Fused=%.1f m  v=%.2f m/s  ±%.1f m",
                ad_buf, tr_buf, est.distance_m, est.velocity_ms, est.distance_stddev_m);
    }

    return est;
}

// ─── Particle filter internals ────────────────────────────────────────────────

void LongitudinalFusion::init_from(float dist_m, float stddev_m)
{
    particles_.resize(static_cast<std::size_t>(cfg_.n_particles));
    std::normal_distribution<float> nd(dist_m, stddev_m);
    std::normal_distribution<float> nv(0.f, 2.f);
    for (auto& p : particles_) {
        p.distance_m  = std::clamp(nd(rng_), 0.f, cfg_.d_max_m);
        p.velocity_ms = nv(rng_);
        p.log_w       = 0.f;   // uniform: log(1/N) omitted, handled in linear_weights()
    }
}

void LongitudinalFusion::predict(float dt_s)
{
    std::normal_distribution<float> nd(0.f, cfg_.process_noise_dist_m);
    std::normal_distribution<float> nv(0.f, cfg_.process_noise_vel_ms);
    for (auto& p : particles_) {
        p.distance_m  = std::clamp(p.distance_m + p.velocity_ms * dt_s + nd(rng_), 0.f, cfg_.d_max_m);
        p.velocity_ms = p.velocity_ms + nv(rng_);
    }
}

float LongitudinalFusion::gaussian_loglik(float z, float mean, float sigma)
{
    const float d = z - mean;
    return -0.5f * (d / sigma) * (d / sigma);
}

void LongitudinalFusion::weight_update(const Meas& ad, const Meas& tr)
{
    // MRPT style: accumulate in log-space — no exp() here, so no underflow.
    // Even if a particle is 80 m from a tight tracker measurement, log_w just
    // becomes a large negative number; it never underflows to exactly 0.
    for (auto& p : particles_) {
        if (ad.valid) p.log_w += gaussian_loglik(ad.distance_m, p.distance_m, ad.stddev_m);
        if (tr.valid) p.log_w += gaussian_loglik(tr.distance_m, p.distance_m, tr.stddev_m);
    }
}

// Log-sum-exp normalisation (same trick MRPT uses in getMean).
// Shifts all log-weights by the maximum before exp() so the largest weight
// becomes 1.0 — impossible to underflow regardless of how negative the others are.
std::vector<float> LongitudinalFusion::linear_weights() const
{
    const std::size_t N = particles_.size();
    float max_lw = particles_[0].log_w;
    for (const auto& p : particles_) max_lw = std::max(max_lw, p.log_w);

    std::vector<float> w(N);
    float sum = 0.f;
    for (std::size_t i = 0; i < N; ++i) {
        w[i] = std::exp(particles_[i].log_w - max_lw);   // largest particle → 1.0
        sum  += w[i];
    }
    if (sum < 1e-12f) {
        const float w0 = 1.f / static_cast<float>(N);
        for (auto& wi : w) wi = w0;
    } else {
        for (auto& wi : w) wi /= sum;
    }
    return w;
}

float LongitudinalFusion::effective_n() const
{
    const auto w = linear_weights();
    float ss = 0.f;
    for (auto wi : w) ss += wi * wi;
    return 1.f / (ss + 1e-12f);
}

void LongitudinalFusion::resample()
{
    const int N = static_cast<int>(particles_.size());
    if (N == 0) return;

    const auto w = linear_weights();

    // Build cumulative sum from normalised linear weights.
    std::vector<float> cs(static_cast<std::size_t>(N));
    cs[0] = w[0];
    for (int i = 1; i < N; ++i)
        cs[static_cast<std::size_t>(i)] = cs[static_cast<std::size_t>(i-1)] + w[static_cast<std::size_t>(i)];

    std::vector<Particle> np;
    np.reserve(static_cast<std::size_t>(N));
    std::uniform_real_distribution<float> u(0.f, 1.f / static_cast<float>(N));
    const float u0 = u(rng_);
    int j = 0;
    for (int i = 0; i < N; ++i) {
        const float thr = u0 + static_cast<float>(i) / static_cast<float>(N);
        while (j < N-1 && cs[static_cast<std::size_t>(j)] < thr) ++j;
        // Reset log_w = 0 after resample so the next frame starts from uniform.
        np.push_back({particles_[static_cast<std::size_t>(j)].distance_m,
                      particles_[static_cast<std::size_t>(j)].velocity_ms,
                      0.f});
    }
    particles_ = std::move(np);
}

}  // namespace visionpilot::fusion
