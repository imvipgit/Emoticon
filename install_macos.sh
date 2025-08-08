#!/bin/bash

# Emoticon Installation Script for macOS
# This script installs all dependencies and sets up the environment

set -e  # Exit on any error

echo "🚀 Starting Emoticon installation for macOS..."

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

# Check if Homebrew is installed
check_homebrew() {
    if command -v brew &> /dev/null; then
        print_success "Homebrew is installed"
        return 0
    else
        print_error "Homebrew is not installed. Please install it first:"
        echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
}

# Install system dependencies using Homebrew
install_system_deps() {
    print_status "Installing system dependencies via Homebrew..."
    
    # Core dependencies
    brew install python@3.11
    brew install opencv
    brew install cmake
    brew install pkg-config
    
    # Additional dependencies
    brew install ffmpeg
    brew install gstreamer
    brew install gst-plugins-base
    brew install gst-plugins-good
    brew install gst-plugins-bad
    brew install gst-plugins-ugly
    
    print_success "System dependencies installed"
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
    
    # Install macOS-compatible requirements
    pip install opencv-python==4.8.1.78
    pip install numpy==1.24.3
    pip install pillow==10.0.1
    
    # Install PyTorch for macOS
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    
    # Install other dependencies
    pip install flask==2.3.3
    pip install flask-cors==4.0.0
    pip install flask-socketio==5.3.6
    pip install scikit-image==0.21.0
    pip install scipy==1.11.1
    pip install scikit-learn==1.3.0
    pip install pyyaml==6.0.1
    pip install requests==2.31.0
    pip install tqdm==4.66.1
    pip install matplotlib==3.7.2
    pip install seaborn==0.12.2
    pip install pytest==7.4.2
    pip install pytest-cov==4.1.0
    pip install black==23.7.0
    pip install flake8==6.0.0
    pip install isort==5.12.0
    
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
    
    # Set executable permissions
    chmod +x src/main.py
    chmod +x install_macos.sh
    
    print_success "Permissions set up"
}

# Create startup script
create_startup_script() {
    print_status "Creating startup script..."
    
    cat > start_emoticon.sh << 'EOF'
#!/bin/bash

# Emoticon Startup Script for macOS
echo "Starting Emoticon..."

# Activate virtual environment
source emoticon_env/bin/activate

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
    echo "    for macOS"
    echo "=========================================="
    echo ""
    
    # Check if we're in the right directory
    if [ ! -f "requirements.txt" ]; then
        print_error "Please run this script from the Emoticon project directory"
        exit 1
    fi
    
    # Check Homebrew
    check_homebrew
    
    # Run installation steps
    install_system_deps
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
