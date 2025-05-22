# Truck Safety: Forward Collision and Lane Merge Warning System

## Project Overview
The **Truck Safety project** is a 2024-25 Senior Design Project (SDP) developed by [Your Name], a Computer Science and Engineering student at UC Irvine. This system enhances truck safety through real-time **Forward Collision Warning (FCW)** and **Lane Merge Warning (LMW)**, leveraging advanced computer vision and edge AI. Built with the **Hailo AI library** and **YOLOv10s** model, it detects vehicles, pedestrians, and lane markings with high accuracy (mAP 0.7–0.8, IoU >0.8) and low latency (<20ms), suitable for edge deployment on devices like the NVIDIA Jetson Orin Nano and Hailo 8L on Raspberry Pi 5.

As the lead developer, I designed a modular pipeline in Python, integrating the Hailo framework for efficient inference on Hailo 8L. Drawing from my experience with **YOLO-NAS**, **HAWQ quantization**, and the **BDD100K dataset**, I aim to extend this project with **C++** components (e.g., lane detection, quantization) to optimize performance for real-time AV systems, aligning with industry needs for scalable, safety-critical software.

## Key Features
- **Forward Collision Warning (FCW)**: Detects vehicles and pedestrians in real-time, warning drivers to prevent collisions (30 FPS on test videos).
- **Lane Merge Warning (LMW)**: Identifies lane markings using classical vision techniques, ensuring safe lane changes (IoU >0.8 on BDD100K).
- **Edge-Optimized Inference**: Utilizes Hailo’s AI accelerator for low-latency object detection with YOLOv10s, achieving <20ms per frame.
- **Modular Design**: Implements object-oriented principles for maintainability, with plans for C++ optimization to enhance performance.

## My Contributions
- **Pipeline Design**: Architected a modular Python pipeline (`TPD_detection.py`) using the Hailo framework, integrating YOLOv10s for object detection and OpenCV for lane processing, demonstrating **object-oriented design (OOD)**.
- **Model Optimization**: Adapted YOLOv10s for edge inference, drawing on my **HAWQ quantization** experience to reduce model size by ~50% while maintaining accuracy (mAP 0.7–0.8).
- **Data Processing**: Preprocessed **BDD100K** subsets (10,000 images) for training and validation, using Python and JSON parsing, with plans to port to C++ for efficiency.
- **C++ Readiness**: Actively developing C++ modules (e.g., lane detection with OpenCV, HAWQ sensitivity analysis with Eigen) to optimize real-time performance, aligning with AV industry standards for embedded systems.
- **Skills Demonstrated**: Python, Hailo, computer vision, OOD, data structures (e.g., `std::vector` in planned C++), and debugging, with transferable skills from UC Irvine coursework and AV research.

## Setup Instructions
To run the Truck Safety system, follow these steps to configure the environment and install dependencies.

### 1. Environmental Configuration
Activate the Hailo virtual environment using the provided script:
```bash
source setup_env.sh
```
If version conflicts occur, install specific Hailo packages:
```bash
sudo apt install hailo-tappas-core=3.30.0-1 hailo-dkms=4.19.0-1 hailort=4.19.0-3
```

### 2. Requirement Installation
In the activated virtual environment, install Python dependencies:
```bash
pip install -r requirements.txt
```

### 3. Resource Download
Download required resources (e.g., YOLOv10s HEF model):
```bash
./download_resources.sh
```

## Running the Code
Run the main pipeline with:
```bash
python basic_pipelines/TPD_detection.py
```

### Options
- `--input rpi`: Use RPi camera input for real-time detection.
- `--input /path/to/file`: Specify a video/image file (e.g., `test01.mp4`).
- `--show-fps`: Display frames per second on the screen.
- `--hef`: Path to the Hailo HEF model (e.g., `yolov10s.hef`).

### Example
```bash
python basic_pipelines/TPD_detection.py --hef resources/yolov10s.hef --input resources/test01.mp4 --show-fps
```

## Future Work
- **C++ Integration**: Port lane detection and preprocessing to C++ using OpenCV and Eigen, optimizing for real-time performance (<10ms latency) on edge devices like the Jetson Orin Nano.
- **HAWQ Quantization**: Implement Hessian-based sensitivity analysis in C++ to further compress YOLOv10s, targeting <1% mAP loss.
- **Sensor Simulation**: Develop a C++ simulator for truck sensor data (camera, LiDAR) using SFML, enhancing testing for safety-critical scenarios.
- **Scalability**: Extend the pipeline to support multi-camera inputs and cloud integration, leveraging Java/C# for backend services.

## Contact and Portfolio
I’m Jeongmoo Yoo, a UC Irvine CSE graduate passionate about AV and embedded systems. Connect with me to discuss this project or software engineering opportunities:
- **GitHub**: [github.com/jungyoo311](https://github.com/jungyoo311)
- **LinkedIn**: [linkedin.com/in/jungyoo-cse](https://linkedin.com/in/yourprofile)
- **Email**: [jeongmy@uci.edu](mailto:jeongmy@uci.edu)

Explore my other AV projects, including **YOLO-NAS** training on **BDD100K** and **HAWQ quantization**, to see my work in optimizing edge AI for safety-critical applications.
