# Drone Detection System using YOLO11

A real-time drone detection system built with YOLO11 (You Only Look Once version 11) for identifying and tracking drones through webcam feed with audio alert functionality.

## Project Overview

This project implements an automated drone detection system capable of real-time detection using computer vision and deep learning techniques. The system processes video frames from a webcam or video file, identifies drones using a trained YOLO11 model, and triggers audio alerts upon detection.

## Technical Architecture

### Core Components

1. **Object Detection Model**: YOLO11 architecture for real-time object detection
2. **Video Processing**: OpenCV for frame capture and image processing
3. **Alert System**: Multi-threaded audio alert mechanism using playsound
4. **Visualization**: Real-time bounding box annotation with confidence scores

### Key Features

- Real-time drone detection from webcam or video file
- Configurable confidence threshold for detection accuracy
- Bounding box visualization with confidence scores
- Audio alert system with non-blocking thread implementation
- Command-line interface for easy configuration
- Object-oriented design for maintainability and extensibility

## Technical Details

### Detection Pipeline

1. **Frame Acquisition**: Captures video frames using OpenCV VideoCapture
2. **Preprocessing**: Frames are passed directly to YOLO11 model
3. **Inference**: YOLO11 performs object detection with configurable confidence threshold
4. **Post-processing**: Filters detections and extracts bounding box coordinates
5. **Visualization**: Annotates frames with bounding boxes and confidence scores
6. **Alert Trigger**: Activates audio alert when drone is detected

### Model Architecture

- **Framework**: Ultralytics YOLO11
- **Input**: RGB images from video stream
- **Output**: Bounding boxes, class labels, and confidence scores
- **Inference Mode**: Real-time prediction with GPU acceleration support

### Detection Parameters

- Primary confidence threshold: 0.6
- Secondary confidence filter: 0.4
- Non-maximum suppression applied internally by YOLO11
- Frame-by-frame processing for real-time performance

## Project Structure

```
drone-detection-yolo11/
│
├── drone_detection.py          # Main Python script with class-based implementation
├── Untitled.ipynb              # Jupyter notebook with exploratory implementation
├── weights/
│   ├── best.pt                 # Trained YOLO11 model weights
│   └── last.pt                 # Last checkpoint from training
├── yolo11n.pt                  # Base YOLO11 nano model
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster inference)
- Webcam or video file for testing

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/All975-rgb/drone-detection-yolo11.git
cd drone-detection-yolo11
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installation:
```bash
python drone_detection.py --help
```

## Usage

### Basic Usage (Webcam)

```bash
python drone_detection.py
```

### Advanced Usage with Parameters

```bash
# Use custom model weights
python drone_detection.py --model weights/best.pt

# Process video file
python drone_detection.py --source path/to/video.mp4

# Adjust confidence threshold
python drone_detection.py --conf 0.7

# Enable audio alerts
python drone_detection.py --alert path/to/alert_sound.mp3

# Combined example
python drone_detection.py --model weights/best.pt --source 0 --conf 0.6 --alert alert.mp3
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | `weights/best.pt` | Path to YOLO11 model weights |
| `--source` | str/int | `0` | Video source (0 for webcam or file path) |
| `--conf` | float | `0.6` | Confidence threshold for detection |
| `--alert` | str | `None` | Path to alert sound file (optional) |

### Jupyter Notebook

For exploratory analysis and testing:
```bash
jupyter notebook Untitled.ipynb
```

## Implementation Details

### DroneDetector Class

The main detection system is implemented as a Python class with the following methods:

- `__init__()`: Initializes detector with configuration parameters
- `load_model()`: Loads YOLO11 model weights
- `initialize_camera()`: Sets up video capture device
- `detect_and_annotate()`: Performs detection and draws bounding boxes
- `play_alert()`: Handles audio alerts in separate thread
- `run()`: Main detection loop
- `cleanup()`: Releases resources

### Detection Logic

```python
# Pseudocode for detection process
for each frame in video_stream:
    results = model.predict(frame, conf=threshold)
    for each detection in results:
        if confidence > 0.4:
            draw_bounding_box(frame, detection)
            if not alert_triggered:
                trigger_audio_alert()
    display(frame)
```

## Model Training

The YOLO11 model was trained on a custom drone dataset. Training details:

- Base model: YOLO11 nano (yolo11n.pt)
- Training framework: Ultralytics
- Output: Optimized weights saved in `weights/best.pt`

## Performance Considerations

- **Frame Rate**: Depends on hardware (GPU vs CPU)
- **Latency**: Real-time processing with minimal delay
- **Memory Usage**: Efficient due to YOLO11 architecture
- **Accuracy**: Configurable via confidence threshold

## Dependencies

Core libraries used in this project:

- `ultralytics`: YOLO11 implementation
- `opencv-python`: Video processing and visualization
- `torch`: Deep learning framework
- `numpy`: Numerical operations
- `playsound`: Audio alert functionality

See `requirements.txt` for complete list with versions.

## Limitations and Future Improvements

### Current Limitations

- Single class detection (drones only)
- Requires good lighting conditions
- Performance depends on hardware capabilities
- Audio alert plays for entire duration (cannot be interrupted)

### Potential Enhancements

1. Multi-class detection (different drone types)
2. Distance estimation using depth perception
3. Drone trajectory tracking and prediction
4. Integration with drone databases for identification
5. Web interface for remote monitoring
6. Model optimization for edge devices
7. Recording functionality for detected events
8. Analytics dashboard for detection statistics

## Technical Challenges and Solutions

### Challenge 1: Alert Loop Prevention
**Problem**: Audio alert playing continuously during detection
**Solution**: Implemented `alert_triggered` flag to ensure single alert per detection event

### Challenge 2: Frame Processing Blocking
**Problem**: Audio playback blocking video processing
**Solution**: Used threading to play alerts in background without interrupting detection loop

### Challenge 3: Confidence Calibration
**Problem**: False positives or missed detections
**Solution**: Implemented dual-threshold system (0.6 for model, 0.4 for display) for balanced accuracy

## Contributing

Contributions are welcome. Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Commit changes with clear messages
4. Submit a pull request with detailed description

## License

This project is open source and available under the MIT License.

## Author

**Alok Kumar**
Email: alok.yottec@gmail.com
GitHub: [All975-rgb](https://github.com/All975-rgb)

## Acknowledgments

- Ultralytics for YOLO11 implementation
- OpenCV community for computer vision tools
- Open source contributors to all dependencies

## References

- [Ultralytics YOLO11 Documentation](https://docs.ultralytics.com/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

**Note**: This project is intended for educational and research purposes. Ensure compliance with local regulations regarding drone detection and surveillance systems.
