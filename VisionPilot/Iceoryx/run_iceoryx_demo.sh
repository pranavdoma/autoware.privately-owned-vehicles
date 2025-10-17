#!/bin/bash
#
# Script to build and run the full Iceoryx publisher/subscriber demo pipeline.
#
# It performs the following steps:
# 1. Builds the publisher and subscriber executables using CMake.
# 2. Starts the Iceoryx router (iox-roudi) in the background.
# 3. Starts the subscriber application in the background.
# 4. Starts the publisher application in the foreground.
# 5. Cleans up all background processes on exit (e.g., via Ctrl+C).
#

# --- Configuration ---
VIDEO_PATH="/home/pranavdoma/Downloads/autoware.privately-owned-vehicles/VisionPilot/ROS2/data/Road_Driving_Scenes_Normal.mp4"
MODEL_PATH="/home/pranavdoma/Downloads/autoware.privately-owned-vehicles/VisionPilot/ROS2/data/models/AutoSpeed_n.onnx"
PRECISION="fp16"
# ---------------------

# Exit immediately if a command exits with a non-zero status.
set -e

# Function to clean up background processes
cleanup() {
    echo -e "\n\nShutting down Iceoryx demo..."
    # Kill all background jobs of this script
    # The negative PID kills the entire process group
    if [ -n "$ROUDI_PID" ]; then
        kill $ROUDI_PID 2>/dev/null && echo "Stopped iox-roudi (PID $ROUDI_PID)."
    fi
    if [ -n "$SUBSCRIBER_PID" ]; then
        kill $SUBSCRIBER_PID 2>/dev/null && echo "Stopped subscriber (PID $SUBSCRIBER_PID)."
    fi
    echo "Cleanup complete."
}

# Trap the EXIT signal to ensure cleanup runs, even on Ctrl+C
trap cleanup EXIT

# --- Main Script ---

# Check if required files exist
if [ ! -f "$VIDEO_PATH" ]; then
    echo "Error: Video file not found: $VIDEO_PATH"
    exit 1
fi
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found: $MODEL_PATH"
    exit 1
fi

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# 1. Start Iceoryx Router (roudi) with custom configuration
echo "--- Starting Iceoryx router (iox-roudi) in the background ---"
# Check if roudi is already running to avoid errors
if ! pgrep -x "iox-roudi" > /dev/null
then
    iox-roudi -c "$SCRIPT_DIR/roudi_config.toml" > /dev/null 2>&1 &
    ROUDI_PID=$!
    # Give roudi a moment to initialize
    sleep 1
    echo "iox-roudi started with PID $ROUDI_PID (using custom config for large frames)."
else
    echo "iox-roudi is already running."
fi
echo ""

# 2. Start the Subscriber in the background
echo "--- Starting subscriber in the background ---"
./build/subscriber &
SUBSCRIBER_PID=$!
echo "Subscriber started with PID $SUBSCRIBER_PID."
echo "An OpenCV window from the subscriber should appear shortly."
echo "Subscriber metrics will be printed to this terminal."
echo ""

# 3. Start the Publisher in the foreground
echo "--- Starting publisher in the foreground ---"
echo "Video: $VIDEO_PATH"
echo "Model: $MODEL_PATH"
echo "Precision: $PRECISION"
echo "--------------------------------------------"
echo "Press Ctrl+C to stop the publisher and clean up all processes."
echo ""

# The script will wait here until the publisher finishes or is interrupted
./build/publisher "$VIDEO_PATH" "$MODEL_PATH" "$PRECISION"

echo "Publisher finished."
