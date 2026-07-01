# Vision Pilot - Open Source L2 ADAS

<p align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)">
        <img src="./Media/VisionPilot_logo.png" alt="VisionPilot" width="100%">
    </picture>
</p>

<div align="center">

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Discord](https://img.shields.io/discord/953808765935816715?label=Autoware%20Discord)](https://discord.com/invite/Q94UsPvReQ)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/autowarefoundation/autoware.privately-owned-vehicles)
![GitHub Repo stars](https://img.shields.io/github/stars/autowarefoundation/autoware.privately-owned-vehicles)

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=whit)
![ROS](https://img.shields.io/badge/ROS-22314E?style=for-the-badge&logo=ROS&logoColor=whit)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/company/the-autoware-foundation)
[![YouTube](https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@autowarefoundation)
[![Website](https://img.shields.io/badge/website-000000?style=for-the-badge&logo=About.me&logoColor=white)](https://autoware.org/)
</div>

<div align="center">

⭐ Star us on GitHub — your support motivates us a lot!

</div>

## Free and fully open-source stack for L2 ADAS
**This codebase contains a productionizable and safety certifiable implementation of an open-source L2 ADAS system called Vision Pilot.**

Vision Pilot is designed to be integrated with automotive OEMs and Tier-1 suppliers in series production passenger cars, and the system can optionally be adopted for transportation and logisitcs use-cases in buses and trucks. 

We offer the complete codebase as free and fully open-source, including AI model weights to help democratize access to this vital technology. Vision Pilot is available under the persmissive Apache 2.0 licence and can freely be used for both commercial and research purposes.

<img src="/Media/hero_GIF.gif" width="100%">

### System Design
<img src="/Media/VisionPilot.png" width="100%">

Vision Pilot is designed to support basic/entry L2 ADAS features for in-lane autonomous driving including the following features:

- **ACC** - autonomous cruise control
- **FCW** - forward collision warning
- **AEB** - autonomous emergency braking
- **LKAS** - lane keep assist
- **LDW** - lane departure warning
- **ISA** - intelligent speed assist
- **Autopilot** - single-lane hands-free highway autopilot

**Sensor specification:**

Vision Pilot can be run with a single, front-facing, monocular camera with 52 - 55 degree horizontal field-of-view, and 2MP resolution.

### Hybrid End-to-End AI Architecture
We utilize a **Hybrid End-to-End AI Architecture** as the core of Vision Pilot in which data is processed in parallel by perception AI models for safety, and End-to-End AI models for performance.

<img src="/Media/VisionPilot_architecture.png" width="100%">

### No Reliance on High Definition Maps
**We do not require 3D high definition maps**. Vision Pilot operates in a 'mapless' mode to follow the road geometry in real-time.

## Getting Started

### How to download and build Vision Pilot

To get started with the project, download the source code from:
```bash
  git clone https://github.com/autowarefoundation/autoware_vision_pilot.git
```
Download ONNX Runtime from the GitHub [releases](https://github.com/microsoft/onnxruntime/releases) page.

Build the project:

```bash
  cd VisionPilot
```

```bash
  mkdir build && cd build
```

```bash
  cmake -DONNXRUNTIME_ROOT=<ONNX_RUNTIME_ROOT_PATH> ../
```

or with ROS2 support 

```bash
  cmake -DONNXRUNTIME_ROOT=<ONNX_RUNTIME_ROOT_PATH> -DENABLE_ROS2_INTERFACE=ON ../
```

```bash
  make
```
This will build the project and create VisionPilot executable inside the build directory.

#### Run Vision Pilot on test data and visualize outputs

**OpenLane Dataset:**

To test Vision Pilot using open loop scenario testing, first download the sample data from the [Google Drive]() directory.

This directory contains video composed of image data from sequences in the appropriate dataset and vehicle speed data 
extracted from the dataset.

Update VisionPilot config files `vision_pilot.conf` inside `config` directory and set: 

```
source.mode             = video
```
and `vision_pilot_test.conf` set:

```
source.input_video         = <INPUT_VIDEO_FILE_PATH>
source.input_vehicle_speed = <INPUT_VEHICLE_SPEED_FILE_PATH>
```
to point to the appropriate video file path and vehicle speed file path.

*Note*: When VisionPilot built from source, update config files before the build. 

Run VisionPilot from inside `build` directory

```
./VisionPilot
```

### Install Vision Pilot from a prebuilt binary

This method is recommended in case new system installation and cuda dependencies are not installed yet.

Download [VisionPilot](https://github.com/autowarefoundation/autoware_vision_pilot.git) prebuilt binary.

Install the .deb package

```bash
  sudo apt install ./VisionPilot-1.0-x86_64.deb
```
Reboot the system, cuda dependencies for VisionPilot installed.

#### Run Vision Pilot on test data and visualize outputs

**OpenLane Dataset:**

To test Vision Pilot using open loop scenario testing, first download the sample data from the [Google Drive]() directory.

This directory contains video composed of image data from sequences in the appropriate dataset and vehicle speed data 
extracted from the dataset.

Update VisionPilot config files `vision_pilot.conf` set: 

```
source.mode             = video
```
and `vision_pilot_test.conf` set:

```
source.input_video         = <INPUT_VIDEO_FILE_PATH>
source.input_vehicle_speed = <INPUT_VEHICLE_SPEED_FILE_PATH>
```
to point to the appropriate video file path and vehicle speed file path.

*Note*: When VisionPilot installed from prebuilt binary update the config files inside: 

```
/usr/share/visionpilot/config
```
directory.

Run VisionPilot from the command line

```
VisionPilot
```

### Coming Soon:

- How to run Vision Pilot with simulators
- How to run Vision Pilot with your own camera on a real vehicle

## Roadmap

- Containerization of Vision Pilot
- Simulation integration guide
- Automotive Hardware, Automotive 
- Real camera and vehicle integration guide
- Support for 8MP camera resolution with 120 degree horizontal field-of-view
- Support for fusion between front-facing camera and automotive RADAR
- Safety Verification and Automotive Standards Compliance (ISO26262, ISO8800)

## Contributing

To learn more about how to participate in this project, please read the [onboarding guide](/ONBOARDING.md)