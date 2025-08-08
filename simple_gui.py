#!/usr/bin/env python3
"""
Simplified GUI for testing camera feed and emotion display
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from camera_manager import CameraManager
from face_detector import FaceDetector
from emotion_detector import EmotionDetector
from utils.preprocessing import ImagePreprocessor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class SimpleEmoticonGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Emoticon - Simple GUI")
        self.root.geometry("1000x600")
        self.root.configure(bg='#2c3e50')
        
        # Variables
        self.is_running = False
        self.camera_manager = None
        self.face_detector = None
        self.emotion_detector = None
        self.preprocessor = None
        
        # Setup GUI
        self.setup_gui()
        
        # Initialize components
        self.initialize_components()
        
        # Protocol for closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def setup_gui(self):
        """Setup the GUI layout"""
        # Main frame
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = tk.Label(
            main_frame,
            text="Emoticon - Emotion Detection",
            font=('Arial', 24, 'bold'),
            bg='#2c3e50',
            fg='#ecf0f1'
        )
        title_label.pack(pady=(0, 20))
        
        # Content frame
        content_frame = tk.Frame(main_frame, bg='#2c3e50')
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left side - Camera feed
        left_frame = tk.Frame(content_frame, bg='#34495e', relief=tk.RAISED, bd=2)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        camera_label = tk.Label(
            left_frame,
            text="Camera Feed",
            font=('Arial', 16, 'bold'),
            bg='#34495e',
            fg='#ecf0f1'
        )
        camera_label.pack(pady=10)
        
        # Video display area
        self.video_label = tk.Label(
            left_frame,
            text="Camera not started",
            font=('Arial', 12),
            bg='#2c3e50',
            fg='#bdc3c7',
            width=50,
            height=15
        )
        self.video_label.pack(pady=20)
        
        # Right side - Emotion display
        right_frame = tk.Frame(content_frame, bg='#34495e', relief=tk.RAISED, bd=2)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        
        emotion_label = tk.Label(
            right_frame,
            text="Emotion Detection",
            font=('Arial', 16, 'bold'),
            bg='#34495e',
            fg='#ecf0f1'
        )
        emotion_label.pack(pady=10)
        
        # Emotion icon
        self.emotion_icon_label = tk.Label(
            right_frame,
            text="😐",
            font=('Arial', 72),
            bg='#34495e',
            fg='#ecf0f1'
        )
        self.emotion_icon_label.pack(pady=20)
        
        # Emotion text
        self.emotion_text_label = tk.Label(
            right_frame,
            text="Neutral",
            font=('Arial', 18, 'bold'),
            bg='#34495e',
            fg='#ecf0f1'
        )
        self.emotion_text_label.pack(pady=10)
        
        # Confidence
        self.confidence_label = tk.Label(
            right_frame,
            text="Confidence: 0%",
            font=('Arial', 14),
            bg='#34495e',
            fg='#bdc3c7'
        )
        self.confidence_label.pack(pady=10)
        
        # Buttons frame
        button_frame = tk.Frame(main_frame, bg='#2c3e50')
        button_frame.pack(pady=20)
        
        # Start button
        self.start_button = tk.Button(
            button_frame,
            text="Start Detection",
            command=self.start_detection,
            font=('Arial', 14, 'bold'),
            bg='#27ae60',
            fg='white',
            relief=tk.RAISED,
            bd=3,
            padx=20,
            pady=10
        )
        self.start_button.pack(side=tk.LEFT, padx=10)
        
        # Stop button
        self.stop_button = tk.Button(
            button_frame,
            text="Stop Detection",
            command=self.stop_detection,
            font=('Arial', 14, 'bold'),
            bg='#e74c3c',
            fg='white',
            relief=tk.RAISED,
            bd=3,
            padx=20,
            pady=10,
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=10)
        
        # Status
        self.status_label = tk.Label(
            main_frame,
            text="Ready to start",
            font=('Arial', 12),
            bg='#2c3e50',
            fg='#ecf0f1'
        )
        self.status_label.pack(pady=10)
        
    def initialize_components(self):
        """Initialize the emotion detection components"""
        try:
            config_dir = Path(__file__).parent / "config"
            
            logger.info(f"Config directory: {config_dir}")
            
            # Initialize camera manager
            self.camera_manager = CameraManager(config_dir / "camera_config.yaml")
            
            # Initialize face detector
            self.face_detector = FaceDetector(config_dir / "model_config.yaml")
            
            # Initialize emotion detector
            self.emotion_detector = EmotionDetector(config_dir / "model_config.yaml")
            
            # Initialize preprocessor
            self.preprocessor = ImagePreprocessor(config_dir / "model_config.yaml")
            
            self.status_label.config(text="Components initialized successfully")
            logger.info("All components initialized successfully")
            
            # Show a test image
            self.show_test_image()
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            messagebox.showerror("Error", f"Failed to initialize components: {e}")
            self.status_label.config(text="Failed to initialize components")
    
    def show_test_image(self):
        """Show a test image in the video label"""
        try:
            # Create a test image
            test_image = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Add some text to the test image
            cv2.putText(test_image, "Camera Ready", (200, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(test_image, "Click 'Start Detection' to begin", (150, 280), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            
            # Convert to PIL and display
            test_image_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(test_image_rgb)
            photo = ImageTk.PhotoImage(pil_image)
            
            self.video_label.configure(image=photo, text="")
            self.video_label.image = photo
            
        except Exception as e:
            logger.error(f"Error showing test image: {e}")
    
    def start_detection(self):
        """Start the emotion detection"""
        if not self.is_running:
            try:
                # Try to start camera
                self.camera_manager.start()
                
                # Test if camera is working
                test_frame = self.camera_manager.get_frame()
                if test_frame is None:
                    raise Exception("Camera not accessible. Please check camera permissions.")
                
                self.is_running = True
                self.start_button.config(state=tk.DISABLED)
                self.stop_button.config(state=tk.NORMAL)
                self.status_label.config(text="Detection started - Camera active")
                
                # Start the detection thread
                self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
                self.detection_thread.start()
                
            except Exception as e:
                logger.error(f"Failed to start detection: {e}")
                messagebox.showerror("Camera Error", 
                    f"Failed to start camera: {e}\n\n"
                    "Please check:\n"
                    "1. Camera permissions in System Preferences\n"
                    "2. Camera is not being used by another application\n"
                    "3. Camera is properly connected")
                self.status_label.config(text="Camera error - check permissions")
    
    def stop_detection(self):
        """Stop the emotion detection"""
        if self.is_running:
            self.is_running = False
            self.camera_manager.stop()
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.status_label.config(text="Detection stopped")
            self.show_test_image()
    
    def detection_loop(self):
        """Main detection loop"""
        frame_count = 0
        start_time = time.time()
        
        logger.info("Detection loop started")
        
        while self.is_running:
            try:
                # Get frame from camera
                frame = self.camera_manager.get_frame()
                if frame is None:
                    logger.warning("No frame received from camera")
                    time.sleep(0.1)
                    continue
                
                frame_count += 1
                logger.info(f"Processing frame {frame_count}")
                
                # Detect faces
                faces = self.face_detector.detect_faces(frame)
                logger.info(f"Detected {len(faces)} faces")
                
                # Process each detected face
                emotions = []
                for face_bbox in faces:
                    # Extract face region
                    face_img = self.preprocessor.extract_face(frame, face_bbox)
                    
                    if face_img is not None:
                        # Preprocess face image
                        processed_face = self.preprocessor.preprocess(face_img)
                        
                        # Detect emotion
                        emotion_result = self.emotion_detector.detect_emotion(processed_face)
                        
                        if emotion_result:
                            emotions.append({
                                'bbox': face_bbox,
                                'emotion': emotion_result['emotion'],
                                'confidence': emotion_result['confidence']
                            })
                
                # Update GUI with results
                self.root.after(0, self.update_gui, frame, emotions, frame_count, start_time)
                
                if frame_count % 30 == 0:  # Log every 30 frames
                    logger.info(f"Processed {frame_count} frames, {len(emotions)} emotions detected")
                
            except Exception as e:
                logger.error(f"Error in detection loop: {e}")
                time.sleep(0.1)
                continue
    
    def update_gui(self, frame, emotions, frame_count, start_time):
        """Update the GUI with detection results"""
        try:
            logger.info(f"Updating GUI with frame {frame_count}, {len(emotions)} emotions")
            
            # Update video frame
            if frame is not None:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Draw face detection boxes
                for emotion in emotions:
                    bbox = emotion['bbox']
                    cv2.rectangle(frame_rgb, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    
                    # Draw emotion label
                    label = f"{emotion['emotion']}: {emotion['confidence']:.2f}"
                    cv2.putText(frame_rgb, label, (bbox[0], bbox[1] - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Resize frame for display
                height, width = frame_rgb.shape[:2]
                max_width = 640
                max_height = 480
                
                if width > max_width or height > max_height:
                    scale = min(max_width / width, max_height / height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
                
                # Convert to PIL Image
                pil_image = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(pil_image)
                
                # Update video label
                self.video_label.configure(image=photo, text="")
                self.video_label.image = photo
                
                logger.info(f"Updated video frame: {frame_rgb.shape}")
                
                # Update emotion display
                if emotions:
                    emotion = emotions[0]  # Show first detected emotion
                    emotion_name = emotion['emotion']
                    confidence = emotion['confidence']
                    
                    # Update emotion icon and text
                    emotion_icons = {
                        'happy': '😊',
                        'sad': '😢',
                        'angry': '😠',
                        'surprise': '😲',
                        'fear': '😨',
                        'disgust': '🤢',
                        'neutral': '😐'
                    }
                    
                    icon = emotion_icons.get(emotion_name, "😐")
                    self.emotion_icon_label.config(text=icon)
                    self.emotion_text_label.config(text=emotion_name.title())
                    self.confidence_label.config(text=f"Confidence: {confidence*100:.1f}%")
                    
                    logger.info(f"Updated emotion: {emotion_name} ({confidence*100:.1f}%)")
                else:
                    # No emotions detected
                    self.emotion_icon_label.config(text="😐")
                    self.emotion_text_label.config(text="No Face")
                    self.confidence_label.config(text="Confidence: 0%")
                
        except Exception as e:
            logger.error(f"Error in update_gui: {e}")
    
    def on_closing(self):
        """Handle window closing"""
        if self.is_running:
            self.stop_detection()
        self.root.destroy()

def main():
    """Main entry point"""
    app = SimpleEmoticonGUI()
    app.root.mainloop()

if __name__ == "__main__":
    main()
