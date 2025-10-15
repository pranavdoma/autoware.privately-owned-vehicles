
#include <cstdint>

// Define a POD (Plain Old Data) struct for a single detection.
// This is necessary for sending data through Iceoryx, which works with raw memory.
struct DetectionPOD {
    float x1, y1, x2, y2;
    float score;
    int32_t class_id;
};

// The main data structure (topic) to be sent over Iceoryx.
// It uses a fixed-size array for simplicity and performance in this demo.
struct FrameDetectionsTopic {
    // Timestamps for latency measurement (e.g., from std::chrono::steady_clock)
    uint64_t capture_timestamp_ns; // When the frame was grabbed from the source
    uint64_t publish_timestamp_ns; // When the packet is sent by the publisher

    // Frame metadata
    uint32_t frame_width;
    uint32_t frame_height;
    uint32_t frame_channels;
    uint64_t frame_data_size; // Actual size of data in the buffer

    // Detection metadata
    uint32_t num_detections;

    // Fixed-size buffers for performance.
    // Ensure these are large enough for your use case.
    static constexpr uint32_t MAX_DETECTIONS = 100;
    static constexpr uint64_t MAX_FRAME_SIZE = 3840 * 2160 * 3; // 4K UHD RGB (24,883,200 bytes)

    DetectionPOD detections[MAX_DETECTIONS];
    uint8_t frame_data[MAX_FRAME_SIZE];
};
