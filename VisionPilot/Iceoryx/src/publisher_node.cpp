
#include "iceoryx_posh/popo/publisher.hpp"
#include "iceoryx_posh/runtime/posh_runtime.hpp"
#include "iceoryx_topic.hpp"

#include "../../common/include/gstreamer_engine.hpp"
#include "../../common/backends/autospeed/tensorrt_engine.hpp"

#include <iostream>
#include <iomanip>
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

    // Performance tracking
    int frame_count = 0;
    long total_inference_us = 0;
    long total_publish_us = 0;

    while (gstreamer.isActive()) {
        auto capture_start_time = now_ns();
        
        // Loan Iceoryx memory FIRST - work directly with shared memory (zero-copy)
        publisher.loan()
            .and_then([&](auto& sample) {
                // A. Capture frame from GStreamer
                cv::Mat gstreamer_frame = gstreamer.getFrame();
                if (gstreamer_frame.empty()) {
                    std::cerr << "Failed to capture frame, skipping." << std::endl;
                    return;
                }

                // B. Create cv::Mat wrapper around loaned shared memory (no copy!)
                sample->frame_width = gstreamer_frame.cols;
                sample->frame_height = gstreamer_frame.rows;
                sample->frame_channels = gstreamer_frame.channels();
                sample->frame_data_size = gstreamer_frame.total() * gstreamer_frame.elemSize();
                
                if (sample->frame_data_size > FrameDetectionsTopic::MAX_FRAME_SIZE) {
                    std::cerr << "Error: Frame size exceeds buffer, skipping." << std::endl;
                    return;
                }

                // Create Mat wrapper pointing to shared memory buffer
                cv::Mat shared_frame(gstreamer_frame.rows, gstreamer_frame.cols, 
                                    CV_8UC3, sample->frame_data);
                
                // Copy from GStreamer to shared memory (unavoidable, but only ONE copy)
                gstreamer_frame.copyTo(shared_frame);

                // C. Run inference directly on shared memory frame (no extra copy!)
                auto infer_start = now_ns();
                std::vector<Detection> detections = backend.inference(shared_frame, conf_thresh, iou_thresh);
                auto infer_end = now_ns();
                long inference_us = (infer_end - infer_start) / 1000;
                total_inference_us += inference_us;

                // D. Store detections and metadata directly in loaned sample
                sample->num_detections = std::min((uint32_t)detections.size(), 
                                                 FrameDetectionsTopic::MAX_DETECTIONS);
                for (uint32_t i = 0; i < sample->num_detections; ++i) {
                    sample->detections[i].x1 = detections[i].x1;
                    sample->detections[i].y1 = detections[i].y1;
                    sample->detections[i].x2 = detections[i].x2;
                    sample->detections[i].y2 = detections[i].y2;
                    sample->detections[i].score = detections[i].confidence;
                    sample->detections[i].class_id = detections[i].class_id;
                }
                
                // E. Set timestamps and publish
                sample->capture_timestamp_ns = capture_start_time;
                sample->publish_timestamp_ns = now_ns();
                sample.publish();

                // Track publish overhead (timestamp setting is negligible)
                long publish_us = (now_ns() - infer_end) / 1000;
                total_publish_us += publish_us;
                frame_count++;
            })
            .or_else([](auto& error) {
                std::cerr << "Failed to loan sample, error: " << error << std::endl;
            });
        
        // Print metrics every 30 frames
        if (frame_count % 30 == 0) {
            std::cout << "\n========================================\n";
            std::cout << "PUBLISHER METRICS - Frame " << frame_count << "\n";
            std::cout << "========================================\n";
            std::cout << "Avg Inference Time:  " << std::fixed << std::setprecision(3) 
                      << (total_inference_us / (double)frame_count / 1000.0) << " ms\n";
            std::cout << "Avg Publish Time:    " << (total_publish_us / (double)frame_count / 1000.0) << " ms\n";
            std::cout << "Detections:          " << detections.size() << "\n";
            std::cout << "========================================\n\n";
        }
    }

    gstreamer.stop();
    std::cout << "\nPublisher stopped." << std::endl;
    std::cout << "Total frames processed: " << frame_count << std::endl;
    return 0;
}
