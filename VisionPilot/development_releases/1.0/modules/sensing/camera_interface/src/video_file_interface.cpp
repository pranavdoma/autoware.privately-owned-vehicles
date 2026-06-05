#include "camera_interface/video_file_interface.hpp"

#include <thread>

namespace camera_interface {

VideoFileInterface::VideoFileInterface(const std::string& path, bool loop, bool realtime)
    : path_(path), loop_(loop), realtime_(realtime)
{
    cap_.open(path);
    if (!cap_.isOpened()) {
        return;
    }

    const double fps = cap_.get(cv::CAP_PROP_FPS);
    if (realtime_ && fps > 1.0) {
        frame_period_ = std::chrono::duration<double>(1.0 / fps);
    }
}

bool VideoFileInterface::is_device_open() const
{
    return cap_.isOpened();
}

std::tuple<bool, cv::Mat> VideoFileInterface::get_latest_frame()
{
    const auto t0 = std::chrono::steady_clock::now();

    cv::Mat frame;
    if (!cap_.read(frame) || frame.empty()) {
        if (!loop_) {
            finished_ = true;
            return {false, {}};
        }
        cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
        rewind_pending_ = true;
        if (!cap_.read(frame) || frame.empty()) {
            return {false, {}};
        }
    }

    if (frame_period_.count() > 0) {
        const auto elapsed = std::chrono::steady_clock::now() - t0;
        const auto rem = frame_period_ - elapsed;
        if (rem.count() > 0) {
            std::this_thread::sleep_for(rem);
        }
    }

    return {true, frame};
}

std::vector<std::string> VideoFileInterface::get_overlay() const
{
    return {"video: " + path_};
}

bool VideoFileInterface::is_finished() const
{
    return finished_;
}

bool VideoFileInterface::take_rewind()
{
    if (!rewind_pending_) {
        return false;
    }
    rewind_pending_ = false;
    return true;
}

std::string VideoFileInterface::source_label() const
{
    return "video";
}

}  // namespace camera_interface
