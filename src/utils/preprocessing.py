#!/usr/bin/env python3
"""
Image Preprocessing Utilities for Emoticon
Handles face extraction and image normalization
"""

import cv2
import numpy as np
import yaml
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """Image preprocessing utilities for emotion recognition"""
    
    def __init__(self, config_path: Path):
        """Initialize preprocessor with configuration"""
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load model configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded preprocessing configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load preprocessing config: {e}")
            raise
    
    def extract_face(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Extract face region from frame using bounding box"""
        try:
            x, y, w, h = bbox
            
            # Ensure coordinates are within frame bounds
            height, width = frame.shape[:2]
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            w = min(w, width - x)
            h = min(h, height - y)
            
            if w <= 0 or h <= 0:
                logger.warning("Invalid bounding box dimensions")
                return None
            
            # Extract face region
            face_region = frame[y:y+h, x:x+w]
            
            if face_region.size == 0:
                logger.warning("Empty face region extracted")
                return None
            
            return face_region
            
        except Exception as e:
            logger.error(f"Error extracting face: {e}")
            return None
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for emotion detection"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Resize to model input size
            input_size = self.config['model']['emotion']['input_size']
            resized = cv2.resize(gray, tuple(input_size))
            
            # Apply normalization if enabled
            if self.config['model']['preprocessing']['normalize']:
                resized = self._normalize_image(resized)
            
            return resized
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image using ImageNet statistics"""
        try:
            # Convert to float32
            image = image.astype(np.float32)
            
            # Normalize to [0, 1]
            image = image / 255.0
            
            # Apply ImageNet normalization if configured
            if self.config['model']['preprocessing']['normalize']:
                mean = self.config['model']['preprocessing']['mean']
                std = self.config['model']['preprocessing']['std']
                
                # For grayscale images, use the first channel values
                image = (image - mean[0]) / std[0]
            
            return image
            
        except Exception as e:
            logger.error(f"Error normalizing image: {e}")
            return image
    
    def augment_image(self, image: np.ndarray) -> np.ndarray:
        """Apply data augmentation to image"""
        try:
            if not self.config['model']['preprocessing']['augment']:
                return image
            
            # Random horizontal flip
            if np.random.random() > 0.5:
                image = cv2.flip(image, 1)
            
            # Random rotation
            if np.random.random() > 0.5:
                angle = np.random.uniform(-10, 10)
                height, width = image.shape[:2]
                center = (width // 2, height // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                image = cv2.warpAffine(image, rotation_matrix, (width, height))
            
            # Random brightness adjustment
            if np.random.random() > 0.5:
                alpha = np.random.uniform(0.8, 1.2)
                beta = np.random.uniform(-10, 10)
                image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            
            return image
            
        except Exception as e:
            logger.error(f"Error augmenting image: {e}")
            return image
    
    def enhance_face(self, face_image: np.ndarray) -> np.ndarray:
        """Enhance face image for better detection"""
        try:
            # Convert to grayscale if needed
            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_image
            
            # Apply histogram equalization
            enhanced = cv2.equalizeHist(gray)
            
            # Apply Gaussian blur to reduce noise
            enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error enhancing face: {e}")
            return face_image
    
    def detect_landmarks(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Detect facial landmarks for advanced preprocessing"""
        try:
            # Convert to grayscale
            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_image
            
            # Initialize face detector
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return None
            
            # Get the largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            
            return face_roi
            
        except Exception as e:
            logger.error(f"Error detecting landmarks: {e}")
            return None
    
    def create_batch(self, images: list) -> np.ndarray:
        """Create a batch of preprocessed images"""
        try:
            preprocessed_images = []
            
            for image in images:
                if image is not None:
                    preprocessed = self.preprocess(image)
                    preprocessed_images.append(preprocessed)
            
            if not preprocessed_images:
                return np.array([])
            
            # Stack images into batch
            batch = np.stack(preprocessed_images)
            
            return batch
            
        except Exception as e:
            logger.error(f"Error creating batch: {e}")
            return np.array([])
    
    def get_preprocessing_info(self) -> Dict[str, Any]:
        """Get preprocessing configuration information"""
        return {
            'input_size': self.config['model']['emotion']['input_size'],
            'normalize': self.config['model']['preprocessing']['normalize'],
            'augment': self.config['model']['preprocessing']['augment'],
            'mean': self.config['model']['preprocessing']['mean'],
            'std': self.config['model']['preprocessing']['std']
        }
