#!/usr/bin/env python3
"""
Camera Manager for Emoticon
Handles camera initialization and frame capture optimized for Jetson hardware
"""

import cv2
import numpy as np
import yaml
import logging
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)

class CameraManager:
    """Manages camera operations optimized for Jetson hardware"""
    
    def __init__(self, config_path: Path):
        """Initialize camera manager with configuration"""
        self.config_path = config_path
        self.config = self._load_config()
        self.camera = None
        self.is_running = False
        self.frame_count = 0
        self.start_time = None
        
        # Jetson specific imports
        try:
            import jetson.utils
            self.jetson_available = True
            logger.info("Jetson utilities available")
        except ImportError:
            self.jetson_available = False
            logger.warning("Jetson utilities not available, using OpenCV fallback")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load camera configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded camera configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load camera config: {e}")
            raise
    
    def start(self):
        """Start the camera"""
        try:
            if self.jetson_available and self._is_csi_camera():
                self._start_csi_camera()
            else:
                self._start_usb_camera()
            
            self.is_running = True
            self.start_time = time.time()
            logger.info("Camera started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start camera: {e}")
            raise
    
    def _is_csi_camera(self) -> bool:
        """Check if CSI camera is available"""
        device = self.config['camera']['device']
        return isinstance(device, str) and device.startswith('/dev/video')
    
    def _start_csi_camera(self):
        """Start CSI camera using Jetson utilities"""
        try:
            import jetson.utils
            
            # Get CSI camera settings
            csi_config = self.config['camera']['csi']
            
            # Create CSI camera
            self.camera = jetson.utils.gstCamera(
                csi_config['width'],
                csi_config['height'],
                f"/dev/video{csi_config['sensor_id']}"
            )
            
            if not self.camera.IsStreaming():
                raise RuntimeError("Failed to start CSI camera stream")
            
            logger.info("CSI camera started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start CSI camera: {e}")
            # Fallback to USB camera
            logger.info("Falling back to USB camera")
            self._start_usb_camera()
    
    def _start_usb_camera(self):
        """Start USB camera using OpenCV"""
        try:
            device = self.config['camera']['device']
            width = self.config['camera']['width']
            height = self.config['camera']['height']
            fps = self.config['camera']['fps']
            
            # Create camera capture
            self.camera = cv2.VideoCapture(device)
            
            if not self.camera.isOpened():
                raise RuntimeError(f"Failed to open camera device {device}")
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.camera.set(cv2.CAP_PROP_FPS, fps)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, self.config['camera']['buffer_size'])
            
            # Set auto exposure and focus if enabled
            if self.config['camera']['auto_exposure']:
                self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            
            if self.config['camera']['auto_focus']:
                self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            
            logger.info(f"USB camera started: {width}x{height} @ {fps}fps")
            
        except Exception as e:
            logger.error(f"Failed to start USB camera: {e}")
            raise
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get current frame from camera"""
        if not self.is_running or self.camera is None:
            return None
        
        try:
            if self.jetson_available and hasattr(self.camera, 'CaptureRGBA'):
                # Jetson CSI camera
                frame = self.camera.CaptureRGBA()
                if frame is not None:
                    # Convert RGBA to BGR
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                    self.frame_count += 1
                    return frame_bgr
            else:
                # USB camera
                ret, frame = self.camera.read()
                if ret:
                    self.frame_count += 1
                    return frame
            
            return None
            
        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            return None
    
    def get_timestamp(self) -> float:
        """Get current timestamp"""
        return time.time()
    
    def get_fps(self) -> float:
        """Calculate current FPS"""
        if self.start_time is None:
            return 0.0
        
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            return self.frame_count / elapsed_time
        return 0.0
    
    def get_resolution(self) -> Tuple[int, int]:
        """Get current camera resolution"""
        if self.camera is None:
            return (0, 0)
        
        if self.jetson_available and hasattr(self.camera, 'GetWidth'):
            return (self.camera.GetWidth(), self.camera.GetHeight())
        else:
            width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (width, height)
    
    def display_frame(self, frame: np.ndarray):
        """Display frame (for debugging)"""
        if frame is not None:
            cv2.imshow('Emoticon - Facial Expression Recognition', frame)
            cv2.waitKey(1)
    
    def stop(self):
        """Stop the camera"""
        self.is_running = False
        
        if self.camera is not None:
            if self.jetson_available and hasattr(self.camera, 'Close'):
                self.camera.Close()
            else:
                self.camera.release()
            
            self.camera = None
        
        cv2.destroyAllWindows()
        logger.info("Camera stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get camera status information"""
        return {
            'running': self.is_running,
            'frame_count': self.frame_count,
            'fps': self.get_fps(),
            'resolution': self.get_resolution(),
            'jetson_available': self.jetson_available,
            'camera_type': 'CSI' if self.jetson_available and self._is_csi_camera() else 'USB'
        }
    
    def __del__(self):
        """Cleanup on deletion"""
        self.stop()
