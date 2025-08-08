#!/usr/bin/env python3
"""
Emoticon - Facial Expression Recognition for NVIDIA Jetson
Main application entry point
"""

import os
import sys
import signal
import logging
import argparse
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

from camera_manager import CameraManager
from face_detector import FaceDetector
from emotion_detector import EmotionDetector
from web_server import WebServer
from utils.preprocessing import ImagePreprocessor
from utils.visualization import Visualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('emoticon.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EmoticonApp:
    """Main application class for facial expression recognition"""
    
    def __init__(self, config_dir="config"):
        """Initialize the application with configuration"""
        self.config_dir = Path(config_dir)
        self.running = False
        self.components = {}
        
        # Initialize components
        self._init_components()
        
    def _init_components(self):
        """Initialize all application components"""
        try:
            # Initialize camera manager
            self.components['camera'] = CameraManager(self.config_dir / "camera_config.yaml")
            
            # Initialize face detector
            self.components['face_detector'] = FaceDetector(self.config_dir / "model_config.yaml")
            
            # Initialize emotion detector
            self.components['emotion_detector'] = EmotionDetector(self.config_dir / "model_config.yaml")
            
            # Initialize preprocessor
            self.components['preprocessor'] = ImagePreprocessor(self.config_dir / "model_config.yaml")
            
            # Initialize visualizer
            self.components['visualizer'] = Visualizer()
            
            # Initialize web server
            self.components['web_server'] = WebServer()
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def start(self):
        """Start the application"""
        try:
            logger.info("Starting Emoticon application...")
            
            # Start camera
            self.components['camera'].start()
            
            # Start web server
            self.components['web_server'].start()
            
            self.running = True
            logger.info("Application started successfully")
            
            # Main processing loop
            self._main_loop()
            
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.error(f"Application error: {e}")
        finally:
            self.stop()
    
    def _main_loop(self):
        """Main processing loop"""
        logger.info("Entering main processing loop...")
        
        while self.running:
            try:
                # Capture frame
                frame = self.components['camera'].get_frame()
                if frame is None:
                    continue
                
                # Detect faces
                faces = self.components['face_detector'].detect_faces(frame)
                
                # Process each detected face
                emotions = []
                for face_bbox in faces:
                    # Extract face region
                    face_img = self.components['preprocessor'].extract_face(frame, face_bbox)
                    
                    if face_img is not None:
                        # Preprocess face image
                        processed_face = self.components['preprocessor'].preprocess(face_img)
                        
                        # Detect emotion
                        emotion_result = self.components['emotion_detector'].detect_emotion(processed_face)
                        
                        if emotion_result:
                            emotions.append({
                                'bbox': face_bbox,
                                'emotion': emotion_result['emotion'],
                                'confidence': emotion_result['confidence']
                            })
                
                # Update web server with results
                self.components['web_server'].update_results({
                    'frame': frame,
                    'emotions': emotions,
                    'timestamp': self.components['camera'].get_timestamp()
                })
                
                # Visualize results (optional)
                if self.components['visualizer'].is_enabled():
                    annotated_frame = self.components['visualizer'].draw_results(frame, emotions)
                    self.components['camera'].display_frame(annotated_frame)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                continue
    
    def stop(self):
        """Stop the application"""
        logger.info("Stopping Emoticon application...")
        self.running = False
        
        # Stop all components
        for name, component in self.components.items():
            try:
                if hasattr(component, 'stop'):
                    component.stop()
                logger.info(f"Stopped {name}")
            except Exception as e:
                logger.error(f"Error stopping {name}: {e}")
        
        logger.info("Application stopped")

def signal_handler(signum, frame):
    """Handle interrupt signals"""
    logger.info(f"Received signal {signum}")
    sys.exit(0)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Emoticon - Facial Expression Recognition")
    parser.add_argument(
        "--config-dir",
        default="config",
        help="Configuration directory path"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    parser.add_argument(
        "--no-web",
        action="store_true",
        help="Disable web interface"
    )
    
    args = parser.parse_args()
    
    # Set debug level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Create and start application
        app = EmoticonApp(args.config_dir)
        app.start()
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
