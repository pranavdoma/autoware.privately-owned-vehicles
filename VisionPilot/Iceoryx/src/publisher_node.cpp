
#include "iceoryx_posh/popo/publisher.hpp"
#include "iceoryx_posh/runtime/posh_runtime.hpp"
#include "iceoryx_topic.hpp"

#include "../../common/include/gstreamer_engine.hpp"
#include "../../common/backends/autospeed/tensorrt_engine.hpp"

#include <iostream>
#include <chrono>

using namespace autoware_pov::vision;
using namespace autoware_pov::vision::autospeed;
using iox::runtime::PoshRuntime;

// Helper to get current time in nanoseconds
uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
}

int main(int argc, char** argv)
{
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <stream_source> <model_path> <precision>\n";
        std::cerr << "  stream_source: RTSP URL, /dev/videoX, or video file\n";
        std::cerr << "  model_path: .pt or .onnx model file\n";
        std::cerr << "  precision: fp32 or fp16\n";
        return 1;
    }

    std::string stream_source = argv[1];
    std::string model_path = argv[2];
    std::string precision = argv[3];
    float conf_thresh = 0.6f;
    float iou_thresh = 0.45f;
    int gpu_id = 0;

    // 1. Initialize GStreamer
    GStreamerEngine gstreamer(stream_source, 0, 0, false); // false = benchmark mode
    if (!gstreamer.initialize() || !gstreamer.start()) {
        std::cerr << "Failed to initialize GStreamer" << std::endl;
        return 1;
    }

    // 2. Initialize TensorRT backend
    AutoSpeedTensorRTEngine backend(model_path, precision, gpu_id);

    // 3. Initialize Iceoryx Runtime and Publisher
    constexpr char APP_NAME[] = "vision-publisher";
    PoshRuntime::initRuntime(APP_NAME);

    iox::popo::PublisherOptions publisherOptions;
    publisherOptions.historyCapacity = 5; // Keep a few samples in history
    iox::popo::Publisher<FrameDetectionsTopic> publisher({"Vision", "AutoSpeed", "Frame"}, publisherOptions);

    std::cout << "Publisher running... " << std::endl;
    std::cout << "Capturing from: " << stream_source << std::endl;
    std::cout << "Publishing to Iceoryx topic 'Vision/AutoSpeed/Frame'" << std::endl;

    while (gstreamer.isActive()) {
        // A. Capture
        auto capture_start_time = now_ns();
        cv::Mat frame = gstreamer.getFrame();
        if (frame.empty()) {
            std::cerr << "Failed to capture frame, stopping." << std::endl;
            break;
        }

        // B. Infer
        std::vector<Detection> detections = backend.inference(frame, conf_thresh, iou_thresh);

        // C. Publish
        publisher.loan()
            .and_then([&](auto& sample) {
                // Get timestamps
                sample->capture_timestamp_ns = capture_start_time;
                sample->publish_timestamp_ns = now_ns();

                // Copy frame data
                sample->frame_width = frame.cols;
                sample->frame_height = frame.rows;
                sample->frame_channels = frame.channels();
                sample->frame_data_size = frame.total() * frame.elemSize();
                
                if (sample->frame_data_size > FrameDetectionsTopic::MAX_FRAME_SIZE) {
                    std::cerr << "Error: Frame size (" << sample->frame_data_size 
                              << " bytes) exceeds max buffer size (" 
                              << FrameDetectionsTopic::MAX_FRAME_SIZE << " bytes)." << std::endl;
                    return;
                }
                memcpy(sample->frame_data, frame.data, sample->frame_data_size);

                // Copy detection data
                sample->num_detections = std::min((uint32_t)detections.size(), FrameDetectionsTopic::MAX_DETECTIONS);
                for (uint32_t i = 0; i < sample->num_detections; ++i) {
                    sample->detections[i] = {
                        detections[i].x1, detections[i].y1, 
                        detections[i].x2, detections[i].y2,
                        detections[i].score,
                        detections[i].class_id
                    };
                }
                
                sample.publish();
            })
            .or_else([](auto& error) {
                std::cerr << "Failed to loan sample, error: " << error << std::endl;
            });
    }

    gstreamer.stop();
    std::cout << "Publisher stopped." << std::endl;
    return 0;
}
