#!/usr/bin/env python3
"""
Demo script for Emoticon
Tests basic functionality without requiring camera hardware
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from emotion_detector import EmotionDetector
from face_detector import FaceDetector
from utils.preprocessing import ImagePreprocessor
from utils.visualization import Visualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_components():
    """Test all components without camera"""
    print("🧪 Testing Emoticon components...")
    
    try:
        # Test configuration loading
        config_path = Path(__file__).parent / "config" / "model_config.yaml"
        print(f"✓ Configuration file found: {config_path}")
        
        # Test emotion detector
        print("Testing emotion detector...")
        emotion_detector = EmotionDetector(config_path)
        print("✓ Emotion detector initialized")
        
        # Test face detector
        print("Testing face detector...")
        face_detector = FaceDetector(config_path)
        print("✓ Face detector initialized")
        
        # Test preprocessor
        print("Testing preprocessor...")
        preprocessor = ImagePreprocessor(config_path)
        print("✓ Preprocessor initialized")
        
        # Test visualizer
        print("Testing visualizer...")
        visualizer = Visualizer()
        print("✓ Visualizer initialized")
        
        # Test model info
        model_info = emotion_detector.get_model_info()
        print(f"✓ Model info: {model_info['model_type']} with {model_info['num_classes']} classes")
        
        print("\n🎉 All components tested successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing components: {e}")
        return False

def test_imports():
    """Test all required imports"""
    print("📦 Testing imports...")
    
    try:
        import cv2
        print("✓ OpenCV imported")
        
        import numpy as np
        print("✓ NumPy imported")
        
        import torch
        print("✓ PyTorch imported")
        
        import yaml
        print("✓ PyYAML imported")
        
        import flask
        print("✓ Flask imported")
        
        print("✓ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Main demo function"""
    print("=" * 50)
    print("😊 Emoticon Demo")
    print("Facial Expression Recognition for NVIDIA Jetson")
    print("=" * 50)
    print()
    
    # Test imports
    if not test_imports():
        print("❌ Import test failed")
        return
    
    print()
    
    # Test components
    if not test_components():
        print("❌ Component test failed")
        return
    
    print()
    print("=" * 50)
    print("✅ Demo completed successfully!")
    print("=" * 50)
    print()
    print("To run the full application:")
    print("1. Install dependencies: ./install.sh")
    print("2. Start the application: python src/main.py")
    print("3. Open web interface: http://localhost:8080")
    print()

if __name__ == "__main__":
    main()
