#!/bin/bash

# Emoticon Installation Script for NVIDIA Jetson
# This script installs all dependencies and sets up the environment

set -e  # Exit on any error

echo "🚀 Starting Emoticon installation for NVIDIA Jetson..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on Jetson
check_jetson() {
    if [ -f "/etc/nv_tegra_release" ]; then
        print_success "Detected NVIDIA Jetson device"
        return 0
    else
        print_warning "Not running on Jetson device - some features may not work optimally"
        return 1
    fi
}

# Update system packages
update_system() {
    print_status "Updating system packages..."
    sudo apt update && sudo apt upgrade -y
    print_success "System packages updated"
}

# Install system dependencies
install_system_deps() {
    print_status "Installing system dependencies..."
    
    # Core dependencies
    sudo apt install -y python3-pip python3-dev python3-venv
    sudo apt install -y libopencv-dev python3-opencv
    sudo apt install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
    sudo apt install -y libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base
    sudo apt install -y gstreamer1.0-plugins-good gstreamer1.0-plugins-bad
    sudo apt install -y gstreamer1.0-plugins-ugly gstreamer1.0-libav
    sudo apt install -y gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa
    sudo apt install -y gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5
    sudo apt install -y gstreamer1.0-pulseaudio
    
    # Additional dependencies
    sudo apt install -y cmake build-essential
    sudo apt install -y libatlas-base-dev
    sudo apt install -y libhdf5-dev libhdf5-serial-dev
    sudo apt install -y libqtgui4 libqtwebkit4 libqt4-test python3-pyqt5
    
    print_success "System dependencies installed"
}

# Install Jetson-specific packages
install_jetson_deps() {
    if check_jetson; then
        print_status "Installing Jetson-specific packages..."
        
        # Install Jetson utilities
        sudo apt install -y jetson-stats
        sudo apt install -y python3-jetson-stats
        
        # Install TensorRT (if available)
        if dpkg -l | grep -q tensorrt; then
            print_success "TensorRT already installed"
        else
            print_warning "TensorRT not found - install manually if needed"
        fi
        
        print_success "Jetson-specific packages installed"
    fi
}

# Create virtual environment
create_venv() {
    print_status "Creating Python virtual environment..."
    
    if [ -d "emoticon_env" ]; then
        print_warning "Virtual environment already exists - removing old one"
        rm -rf emoticon_env
    fi
    
    python3 -m venv emoticon_env
    print_success "Virtual environment created"
}

# Activate virtual environment and install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    
    source emoticon_env/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install dependencies
    pip install -r requirements.txt
    
    print_success "Python dependencies installed"
}

# Download pre-trained models (placeholder)
download_models() {
    print_status "Setting up model directory..."
    
    mkdir -p models
    
    # Create placeholder model files (in real deployment, these would be downloaded)
    print_warning "Creating placeholder model files..."
    echo "Placeholder emotion model" > models/emotion_model.pth
    echo "Placeholder face detection model" > models/face_detection_model.pth
    
    print_success "Model directory set up"
}

# Set up permissions
setup_permissions() {
    print_status "Setting up permissions..."
    
    # Add user to video group for camera access
    sudo usermod -a -G video $USER
    
    # Set executable permissions
    chmod +x src/main.py
    chmod +x install.sh
    
    print_success "Permissions set up"
}

# Create startup script
create_startup_script() {
    print_status "Creating startup script..."
    
    cat > start_emoticon.sh << 'EOF'
#!/bin/bash

# Emoticon Startup Script
echo "Starting Emoticon..."

# Activate virtual environment
source emoticon_env/bin/activate

# Set Jetson performance mode (if on Jetson)
if [ -f "/etc/nv_tegra_release" ]; then
    echo "Setting Jetson performance mode..."
    sudo nvpmodel -m 0
    sudo jetson_clocks
fi

# Start the application
python src/main.py
EOF
    
    chmod +x start_emoticon.sh
    print_success "Startup script created"
}

# Test installation
test_installation() {
    print_status "Testing installation..."
    
    source emoticon_env/bin/activate
    
    # Test Python imports
    python3 -c "
import cv2
import numpy as np
import torch
import yaml
import flask
print('All core dependencies imported successfully')
"
    
    if [ $? -eq 0 ]; then
        print_success "Installation test passed"
    else
        print_error "Installation test failed"
        exit 1
    fi
}

# Main installation function
main() {
    echo "=========================================="
    echo "    Emoticon Installation Script"
    echo "    Facial Expression Recognition"
    echo "    for NVIDIA Jetson"
    echo "=========================================="
    echo ""
    
    # Check if running as root
    if [ "$EUID" -eq 0 ]; then
        print_error "Please do not run this script as root"
        exit 1
    fi
    
    # Check if we're in the right directory
    if [ ! -f "requirements.txt" ]; then
        print_error "Please run this script from the Emoticon project directory"
        exit 1
    fi
    
    # Run installation steps
    update_system
    install_system_deps
    install_jetson_deps
    create_venv
    install_python_deps
    download_models
    setup_permissions
    create_startup_script
    test_installation
    
    echo ""
    echo "=========================================="
    print_success "Installation completed successfully!"
    echo "=========================================="
    echo ""
    echo "To start Emoticon, run:"
    echo "  ./start_emoticon.sh"
    echo ""
    echo "Or manually:"
    echo "  source emoticon_env/bin/activate"
    echo "  python src/main.py"
    echo ""
    echo "Web interface will be available at: http://localhost:8080"
    echo ""
}

# Run main function
main "$@"
