# Emoticon - Facial Expression Recognition for macOS

A real-time facial expression recognition system adapted for macOS using deep learning and computer vision techniques.

## Features

- Real-time facial expression detection
- Support for 7 basic emotions: Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral
- Web-based interface for easy interaction
- REST API for integration with other systems
- High accuracy using pre-trained deep learning models
- Optimized for macOS with CPU-based inference

## Prerequisites

- macOS 10.15+ (Catalina or later)
- Python 3.9+
- Homebrew (for system dependencies)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/vipul-sindha/Emoticon.git
cd Emoticon
```

### 2. Run the Installation Script

```bash
chmod +x install_macos.sh
./install_macos.sh
```

This script will:
- Install system dependencies via Homebrew
- Create a Python virtual environment
- Install all Python dependencies
- Set up placeholder model files
- Create startup scripts

### 3. Test the Installation

```bash
source emoticon_env/bin/activate
python demo.py
```

This will run a comprehensive test of all components.

## Usage

### Starting the Application

```bash
# Option 1: Use the startup script
./start_emoticon.sh

# Option 2: Manual startup
source emoticon_env/bin/activate
python src/main.py
```

### Web Interface

Once the application is running, open your browser and navigate to:
- **Main Interface**: http://localhost:8080
- **API Documentation**: http://localhost:8080/api/docs

### API Usage

The application provides a REST API for integration:

```bash
# Get current emotion
curl http://localhost:8080/api/emotion

# Get emotion history
curl http://localhost:8080/api/emotions/history

# Get system status
curl http://localhost:8080/api/status
```

## Camera Permissions

On macOS, the application requires camera permissions to function properly:

1. When you first run the application, macOS will prompt for camera access
2. Click "Allow" to grant camera permissions
3. If you accidentally denied access, you can enable it later:
   - Go to System Preferences > Security & Privacy > Privacy > Camera
   - Add your terminal application (Terminal.app or iTerm2) to the list

## Configuration

### Camera Configuration

Edit `config/camera_config.yaml` to customize camera settings:

```yaml
camera:
  device: 0                    # Camera device index
  width: 640                   # Frame width
  height: 480                  # Frame height
  fps: 30                      # Frames per second
  codec: "MJPG"               # Video codec
```

### Model Configuration

Edit `config/model_config.yaml` to customize model settings:

```yaml
model:
  emotion_model_path: "models/emotion_model.pth"
  face_detection_model_path: "models/face_detection_model.pth"
  confidence_threshold: 0.7
  gpu_acceleration: false      # Set to false for CPU-only
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
├── demo.py                     # Demo script for testing
├── install_macos.sh           # macOS installation script
└── README_macos.md            # This file
```

## Development

### Running Tests

```bash
# Run the demo script
python demo.py

# Run specific tests
python -m pytest tests/
```

### Code Style

This project follows PEP 8 style guidelines:

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

1. **Camera not working**: 
   - Check camera permissions in System Preferences
   - Ensure your terminal app has camera access

2. **Import errors**:
   - Make sure you're in the virtual environment: `source emoticon_env/bin/activate`
   - Reinstall dependencies: `pip install -r requirements.txt`

3. **Model loading errors**:
   - The current setup uses placeholder models
   - For production use, download actual trained models

4. **Performance issues**:
   - The system uses CPU inference by default
   - For better performance, consider using a GPU-enabled version

### Performance Tips

- Use a good quality webcam for better results
- Ensure good lighting conditions
- Close other applications to free up system resources
- Consider using a dedicated GPU if available

## Differences from Jetson Version

This macOS version differs from the original Jetson version in several ways:

- **CPU-only inference**: No GPU acceleration (PyTorch CPU version)
- **Simplified camera handling**: Uses OpenCV instead of Jetson-specific utilities
- **No TensorRT optimization**: Uses standard PyTorch models
- **Web interface**: Same Flask-based interface as Jetson version
- **API compatibility**: Same REST API endpoints

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support and questions:
- Create an issue on GitHub
- Check the [documentation](docs/)
- Review the troubleshooting section above
