# Emoticon - Facial Expression Recognition for NVIDIA Jetson

A real-time facial expression recognition system optimized for NVIDIA Jetson hardware using deep learning and computer vision techniques.

## Features

- Real-time facial expression detection
- Support for 7 basic emotions: Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral
- Optimized for NVIDIA Jetson Nano/Xavier/Orin
- Web-based interface for easy interaction
- REST API for integration with other systems
- High accuracy using pre-trained deep learning models

## Hardware Requirements

- NVIDIA Jetson Nano/Xavier/Orin
- USB Camera or CSI Camera
- At least 4GB RAM (8GB recommended)
- MicroSD card with at least 16GB storage

## Software Requirements

- JetPack 4.6+ or JetPack 5.0+
- Python 3.8+
- OpenCV 4.5+
- TensorFlow 2.x or PyTorch 1.x
- CUDA 10.2+ (for GPU acceleration)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/vipul-sindha/Emoticon.git
cd Emoticon
```

### 2. Install Dependencies

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y python3-pip python3-dev python3-venv
sudo apt install -y libopencv-dev python3-opencv
sudo apt install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
sudo apt install -y libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base
sudo apt install -y gstreamer1.0-plugins-good gstreamer1.0-plugins-bad
sudo apt install -y gstreamer1.0-plugins-ugly gstreamer1.0-libav
sudo apt install -y gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa
sudo apt install -y gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5
sudo apt install -y gstreamer1.0-pulseaudio

# Create virtual environment
python3 -m venv emoticon_env
source emoticon_env/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Download Pre-trained Models

```bash
# Download emotion recognition model
wget https://github.com/vipul-sindha/Emoticon/releases/download/v1.0/emotion_model.pth -O models/emotion_model.pth

# Download face detection model
wget https://github.com/vipul-sindha/Emoticon/releases/download/v1.0/face_detection_model.pth -O models/face_detection_model.pth
```

### 4. Configure Camera

Edit `config/camera_config.yaml` to set your camera parameters:

```yaml
camera:
  device: 0  # USB camera index or CSI camera path
  width: 640
  height: 480
  fps: 30
  codec: "MJPG"
```

## Usage

### 1. Start the Application

```bash
# Activate virtual environment
source emoticon_env/bin/activate

# Run the main application
python src/main.py
```

### 2. Web Interface

Open your browser and navigate to `http://localhost:8080` to access the web interface.

### 3. API Usage

The application provides a REST API for integration:

```bash
# Get current emotion
curl http://localhost:8080/api/emotion

# Get emotion history
curl http://localhost:8080/api/emotions/history

# Get system status
curl http://localhost:8080/api/status
```

## Project Structure

```
Emoticon/
├── src/
│   ├── main.py                 # Main application entry point
│   ├── emotion_detector.py     # Emotion recognition module
│   ├── face_detector.py        # Face detection module
│   ├── camera_manager.py       # Camera interface
│   ├── web_server.py          # Web server and API
│   └── utils/
│       ├── preprocessing.py    # Image preprocessing utilities
│       └── visualization.py    # Visualization utilities
├── models/
│   ├── emotion_model.pth       # Pre-trained emotion model
│   └── face_detection_model.pth # Face detection model
├── config/
│   ├── camera_config.yaml      # Camera configuration
│   └── model_config.yaml       # Model configuration
├── data/
│   ├── training/               # Training data
│   └── validation/             # Validation data
├── tests/
│   ├── test_emotion_detector.py
│   └── test_face_detector.py
├── docs/
│   ├── api.md                  # API documentation
│   └── deployment.md           # Deployment guide
├── requirements.txt            # Python dependencies
├── setup.py                   # Package setup
└── README.md                  # This file
```

## Configuration

### Camera Configuration

Edit `config/camera_config.yaml`:

```yaml
camera:
  device: 0                    # Camera device index
  width: 640                   # Frame width
  height: 480                  # Frame height
  fps: 30                      # Frames per second
  codec: "MJPG"               # Video codec
  buffer_size: 1              # Buffer size
```

### Model Configuration

Edit `config/model_config.yaml`:

```yaml
model:
  emotion_model_path: "models/emotion_model.pth"
  face_detection_model_path: "models/face_detection_model.pth"
  confidence_threshold: 0.7
  gpu_acceleration: true
  batch_size: 1
```

## Performance Optimization

### Jetson Nano Optimization

```bash
# Set performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

# Enable GPU memory
sudo sh -c 'echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo'
```

### Jetson Xavier/Orin Optimization

```bash
# Set maximum performance
sudo nvpmodel -m 0
sudo jetson_clocks

# Optimize GPU memory
sudo sh -c 'echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo'
```

## Development

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_emotion_detector.py
```

### Code Style

This project follows PEP 8 style guidelines. Use the provided linting tools:

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run linting
flake8 src/
black src/
isort src/
```

## Troubleshooting

### Common Issues

1. **Camera not detected**: Check camera permissions and device index
2. **Low FPS**: Reduce resolution or disable GPU acceleration
3. **Memory issues**: Reduce batch size or model complexity
4. **Model loading errors**: Verify model file paths and permissions

### Performance Tips

- Use CSI camera for better performance
- Enable GPU acceleration when available
- Reduce frame resolution for higher FPS
- Use optimized models for Jetson hardware

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- NVIDIA for Jetson platform
- OpenCV community for computer vision tools
- PyTorch/TensorFlow communities for deep learning frameworks

## Support

For support and questions:
- Create an issue on GitHub
- Check the [documentation](docs/)
- Review the [troubleshooting guide](docs/troubleshooting.md)

## Version History

- v1.0.0 - Initial release with basic emotion recognition
- v1.1.0 - Added web interface and REST API
- v1.2.0 - Performance optimizations for Jetson hardware
