
#include "iceoryx_posh/popo/subscriber.hpp"
#include "iceoryx_posh/runtime/posh_runtime.hpp"
#include "iceoryx_topic.hpp"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <atomic>

using iox::runtime::PoshRuntime;

// Helper to get current time in nanoseconds
uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
}

// Visualization function (from infer_stream.cpp)
void drawDetections(cv::Mat& frame, const std::vector<DetectionPOD>& detections)
{
    auto getColor = [](int class_id) -> cv::Scalar {
        switch(class_id) {
            case 1: return cv::Scalar(0, 0, 255);    // Red
            case 2: return cv::Scalar(0, 255, 255);  // Yellow
            case 3: return cv::Scalar(255, 255, 0);  // Cyan
            default: return cv::Scalar(255, 255, 255); // White
        }
    };

    for (const auto& det : detections) {
        cv::Scalar color = getColor(det.class_id);
        cv::rectangle(frame, 
                     cv::Point(static_cast<int>(det.x1), static_cast<int>(det.y1)), 
                     cv::Point(static_cast<int>(det.x2), static_cast<int>(det.y2)), 
                     color, 2);
    }
}


int main(int argc, char** argv)
{
    // 1. Initialize Iceoryx Runtime and Subscriber
    constexpr char APP_NAME[] = "vision-subscriber";
    PoshRuntime::initRuntime(APP_NAME);

    iox::popo::SubscriberOptions subscriberOptions;
    subscriberOptions.queueCapacity = 5;
    subscriberOptions.historyRequest = 0; // Don't care about history
    iox::popo::Subscriber<FrameDetectionsTopic> subscriber({"Vision", "AutoSpeed", "Frame"}, subscriberOptions);

    std::cout << "Subscriber running..." << std::endl;
    std::cout << "Subscribed to Iceoryx topic 'Vision/AutoSpeed/Frame'" << std::endl;
    std::cout << "Press 'q' in the window to quit." << std::endl;

    cv::namedWindow("Iceoryx Subscriber", cv::WINDOW_NORMAL);
    cv::resizeWindow("Iceoryx Subscriber", 960, 540);

    // For metrics
    std::atomic<long> total_ipc_us{0};           // IPC overhead: publish → receive
    std::atomic<long> total_e2e_us{0};           // End-to-end: capture → receive
    std::atomic<long> total_visualization_us{0}; // Time to draw and display
    std::atomic<int> frame_count{0};
    std::atomic<bool> running{true};

    while (running.load())
    {
        subscriber.take()
            .and_then([&](const auto& sample) {
                auto reception_time_ns = now_ns();

                // A. Calculate latencies
                long ipc_us = (reception_time_ns - sample->publish_timestamp_ns) / 1000;
                long e2e_us = (reception_time_ns - sample->capture_timestamp_ns) / 1000;
                total_ipc_us.fetch_add(ipc_us);
                total_e2e_us.fetch_add(e2e_us);
                int count = frame_count.fetch_add(1) + 1;

                // B. Create cv::Mat wrapper around shared memory (zero-copy!)
                auto viz_start = now_ns();
                cv::Mat frame(sample->frame_height, sample->frame_width, CV_8UC3, (void*)sample->frame_data);
                
                // Wrap detections array (no copy until visualization needs it)
                std::vector<DetectionPOD> detections;
                detections.assign(sample->detections, sample->detections + sample->num_detections);

                // C. Visualize directly on shared memory frame
                drawDetections(frame, detections);
                cv::imshow("Iceoryx Subscriber", frame);

                if (cv::waitKey(1) == 'q') {
                    running.store(false);
                }
                
                auto viz_end = now_ns();
                long viz_us = (viz_end - viz_start) / 1000;
                total_visualization_us.fetch_add(viz_us);

                // D. Print metrics periodically
                if (count % 30 == 0) {
                    long avg_ipc = total_ipc_us.load() / count;
                    long avg_e2e = total_e2e_us.load() / count;
                    long avg_viz = total_visualization_us.load() / count;
                    
                    std::cout << "\n========================================\n";
                    std::cout << "ICEORYX BENCHMARK - Frame " << count << "\n";
                    std::cout << "========================================\n";
                    std::cout << "IPC Latency (publish → receive): " << std::fixed << std::setprecision(3) 
                              << (avg_ipc / 1000.0) << " ms\n";
                    std::cout << "End-to-End (capture → receive):  " << (avg_e2e / 1000.0) << " ms\n";
                    std::cout << "Visualization (draw + display):  " << (avg_viz / 1000.0) << " ms\n";
                    std::cout << "Detections in frame:             " << sample->num_detections << "\n";
                    std::cout << "========================================\n\n";
                }
            })
            .or_else([&](auto& result) {
                if (result != iox::popo::ChunkReceiveResult::NO_CHUNK_AVAILABLE) {
                    std::cerr << "Error receiving chunk: " << result << std::endl;
                }
                // To prevent a busy-wait loop, sleep briefly if no chunk is available
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            });
    }

    cv::destroyAllWindows();
    std::cout << "Subscriber stopped." << std::endl;
    return 0;
}
