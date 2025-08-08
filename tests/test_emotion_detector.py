#!/usr/bin/env python3
"""
Tests for Emotion Detector
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from emotion_detector import EmotionDetector, EmotionCNN

class TestEmotionCNN:
    """Test cases for EmotionCNN model"""
    
    def test_model_creation(self):
        """Test model creation with different number of classes"""
        model = EmotionCNN(num_classes=7)
        assert model is not None
        assert isinstance(model, EmotionCNN)
    
    def test_model_forward_pass(self):
        """Test forward pass through the model"""
        model = EmotionCNN(num_classes=7)
        batch_size = 4
        channels = 1
        height, width = 48, 48
        
        # Create dummy input
        x = torch.randn(batch_size, channels, height, width)
        
        # Forward pass
        output = model(x)
        
        # Check output shape
        assert output.shape == (batch_size, 7)
    
    def test_model_parameters(self):
        """Test that model has trainable parameters"""
        model = EmotionCNN(num_classes=7)
        
        # Check that model has parameters
        params = list(model.parameters())
        assert len(params) > 0
        
        # Check that parameters are trainable
        for param in params:
            assert param.requires_grad

class TestEmotionDetector:
    """Test cases for EmotionDetector class"""
    
    @pytest.fixture
    def config_path(self):
        """Create a temporary config file for testing"""
        config_content = """
model:
  emotion_model_path: "models/emotion_model.pth"
  face_detection_model_path: "models/face_detection_model.pth"
  
  emotion:
    model_type: "cnn"
    input_size: [48, 48]
    num_classes: 7
    classes: ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    
  face_detection:
    model_type: "haar"
    scale_factor: 1.1
    min_neighbors: 5
    min_size: [30, 30]
    
  performance:
    confidence_threshold: 0.7
    gpu_acceleration: true
    batch_size: 1
    num_workers: 2
    
  preprocessing:
    normalize: true
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
    augment: false
"""
        
        config_file = Path(__file__).parent / "test_config.yaml"
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        yield config_file
        
        # Cleanup
        if config_file.exists():
            config_file.unlink()
    
    def test_initialization(self, config_path):
        """Test EmotionDetector initialization"""
        detector = EmotionDetector(config_path)
        assert detector is not None
        assert detector.model is not None
        assert len(detector.emotion_classes) == 7
    
    def test_device_selection(self, config_path):
        """Test device selection logic"""
        detector = EmotionDetector(config_path)
        
        # Device should be either CPU or CUDA
        assert str(detector.device) in ['cpu', 'cuda']
    
    def test_preprocess_image(self, config_path):
        """Test image preprocessing"""
        detector = EmotionDetector(config_path)
        
        # Create test image
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Preprocess image
        processed = detector.preprocess_image(test_image)
        
        # Check output is tensor
        assert isinstance(processed, torch.Tensor)
        
        # Check shape (batch_size=1, channels=1, height=48, width=48)
        assert processed.shape == (1, 1, 48, 48)
    
    def test_detect_emotion_no_face(self, config_path):
        """Test emotion detection with no face"""
        detector = EmotionDetector(config_path)
        
        # Create empty image
        empty_image = np.zeros((48, 48), dtype=np.uint8)
        
        # Detect emotion
        result = detector.detect_emotion(empty_image)
        
        # Should return None for no face
        assert result is None
    
    def test_get_emotion_probabilities(self, config_path):
        """Test getting emotion probabilities"""
        detector = EmotionDetector(config_path)
        
        # Create test image
        test_image = np.random.randint(0, 255, (48, 48), dtype=np.uint8)
        
        # Get probabilities
        probs = detector.get_emotion_probabilities(test_image)
        
        # Check that probabilities sum to 1
        if probs is not None:
            total_prob = sum(probs.values())
            assert abs(total_prob - 1.0) < 1e-6
    
    def test_model_info(self, config_path):
        """Test getting model information"""
        detector = EmotionDetector(config_path)
        
        info = detector.get_model_info()
        
        assert 'model_type' in info
        assert 'num_classes' in info
        assert 'input_size' in info
        assert 'emotion_classes' in info
        assert 'device' in info
        assert 'confidence_threshold' in info

if __name__ == "__main__":
    pytest.main([__file__])
