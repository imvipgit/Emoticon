#!/usr/bin/env python3
"""
Face Detector for Emoticon
Handles face detection using multiple methods optimized for Jetson hardware
"""

import cv2
import numpy as np
import yaml
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

class FaceDetector:
    """Face detection class with multiple detection methods"""
    
    def __init__(self, config_path: Path):
        """Initialize face detector with configuration"""
        self.config_path = config_path
        self.config = self._load_config()
        self.detector = None
        self.face_cascade = None
        
        # Initialize detection method
        self._init_detector()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load model configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded model configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load model config: {e}")
            raise
    
    def _init_detector(self):
        """Initialize the face detection method"""
        detection_config = self.config['model']['face_detection']
        model_type = detection_config['model_type']
        
        try:
            if model_type == 'haar':
                self._init_haar_detector()
            elif model_type == 'dnn':
                self._init_dnn_detector()
            elif model_type == 'mtcnn':
                self._init_mtcnn_detector()
            else:
                logger.warning(f"Unknown detection method: {model_type}, falling back to Haar")
                self._init_haar_detector()
                
            logger.info(f"Initialized {model_type} face detector")
            
        except Exception as e:
            logger.error(f"Failed to initialize {model_type} detector: {e}")
            logger.info("Falling back to Haar cascade")
            self._init_haar_detector()
    
    def _init_haar_detector(self):
        """Initialize Haar cascade face detector"""
        try:
            # Load Haar cascade classifier
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.face_cascade.empty():
                raise RuntimeError("Failed to load Haar cascade classifier")
            
            self.detector_type = 'haar'
            logger.info("Haar cascade detector initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Haar detector: {e}")
            raise
    
    def _init_dnn_detector(self):
        """Initialize DNN-based face detector"""
        try:
            # Load DNN face detection model
            model_path = self.config['model']['face_detection_model_path']
            
            # Use OpenCV's DNN face detector
            self.detector = cv2.dnn.readNetFromCaffe(
                str(Path(model_path).parent / "deploy.prototxt"),
                str(Path(model_path).parent / "res10_300x300_ssd_iter_140000.caffemodel")
            )
            
            self.detector_type = 'dnn'
            logger.info("DNN face detector initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize DNN detector: {e}")
            raise
    
    def _init_mtcnn_detector(self):
        """Initialize MTCNN face detector"""
        try:
            # Import MTCNN
            from mtcnn import MTCNN
            
            self.detector = MTCNN(
                min_face_size=20,
                scale_factor=0.709,
                steps_threshold=[0.6, 0.7, 0.7]
            )
            
            self.detector_type = 'mtcnn'
            logger.info("MTCNN face detector initialized")
            
        except ImportError:
            logger.warning("MTCNN not available, falling back to Haar")
            self._init_haar_detector()
        except Exception as e:
            logger.error(f"Failed to initialize MTCNN detector: {e}")
            raise
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in the given frame"""
        if frame is None:
            return []
        
        try:
            if self.detector_type == 'haar':
                return self._detect_faces_haar(frame)
            elif self.detector_type == 'dnn':
                return self._detect_faces_dnn(frame)
            elif self.detector_type == 'mtcnn':
                return self._detect_faces_mtcnn(frame)
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            return []
    
    def _detect_faces_haar(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using Haar cascade"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            detection_config = self.config['model']['face_detection']
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=detection_config['scale_factor'],
                minNeighbors=detection_config['min_neighbors'],
                minSize=tuple(detection_config['min_size'])
            )
            
            # Convert to list of (x, y, w, h) tuples
            return [(x, y, w, h) for (x, y, w, h) in faces]
            
        except Exception as e:
            logger.error(f"Error in Haar face detection: {e}")
            return []
    
    def _detect_faces_dnn(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using DNN"""
        try:
            # Prepare input blob
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)),
                1.0,
                (300, 300),
                (104.0, 177.0, 123.0)
            )
            
            # Forward pass
            self.detector.setInput(blob)
            detections = self.detector.forward()
            
            # Process detections
            faces = []
            height, width = frame.shape[:2]
            confidence_threshold = self.config['model']['performance']['confidence_threshold']
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                if confidence > confidence_threshold:
                    # Get bounding box coordinates
                    box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                    x, y, w, h = box.astype(int)
                    
                    # Ensure coordinates are within frame bounds
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, width - x)
                    h = min(h, height - y)
                    
                    if w > 0 and h > 0:
                        faces.append((x, y, w, h))
            
            return faces
            
        except Exception as e:
            logger.error(f"Error in DNN face detection: {e}")
            return []
    
    def _detect_faces_mtcnn(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using MTCNN"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            detections = self.detector.detect_faces(rgb_frame)
            
            # Extract bounding boxes
            faces = []
            confidence_threshold = self.config['model']['performance']['confidence_threshold']
            
            for detection in detections:
                if detection['confidence'] > confidence_threshold:
                    x, y, w, h = detection['box']
                    faces.append((x, y, w, h))
            
            return faces
            
        except Exception as e:
            logger.error(f"Error in MTCNN face detection: {e}")
            return []
    
    def draw_faces(self, frame: np.ndarray, faces: List[Tuple[int, int, int, int]], 
                   color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
        """Draw detected faces on the frame"""
        annotated_frame = frame.copy()
        
        for (x, y, w, h) in faces:
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, thickness)
        
        return annotated_frame
    
    def get_detection_info(self) -> Dict[str, Any]:
        """Get information about the current detection method"""
        return {
            'detector_type': self.detector_type,
            'confidence_threshold': self.config['model']['performance']['confidence_threshold'],
            'min_face_size': self.config['model']['face_detection']['min_size'],
            'scale_factor': self.config['model']['face_detection']['scale_factor'],
            'min_neighbors': self.config['model']['face_detection']['min_neighbors']
        }
    
    def __del__(self):
        """Cleanup on deletion"""
        pass
