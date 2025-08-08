#!/usr/bin/env python3
"""
Emoticon GUI Application
Standalone Python GUI for facial expression recognition
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
import threading
import time
from PIL import Image, ImageTk
import logging
from pathlib import Path
import sys

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

from camera_manager import CameraManager
from face_detector import FaceDetector
from emotion_detector import EmotionDetector
from utils.preprocessing import ImagePreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmoticonGUI:
    """Main GUI application for Emoticon"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Emoticon - Facial Expression Recognition")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')
        
        # Initialize components
        self.camera_manager = None
        self.face_detector = None
        self.emotion_detector = None
        self.preprocessor = None
        
        # GUI variables
        self.is_running = False
        self.current_frame = None
        self.current_emotion = "neutral"
        self.current_confidence = 0.0
        
        # Emotion colors and icons
        self.emotion_colors = {
            'happy': '#f39c12',
            'sad': '#3498db',
            'angry': '#e74c3c',
            'fear': '#9b59b6',
            'surprise': '#e67e22',
            'disgust': '#27ae60',
            'neutral': '#95a5a6'
        }
        
        self.emotion_icons = {
            'happy': '😊',
            'sad': '😢',
            'angry': '😠',
            'fear': '😨',
            'surprise': '😲',
            'disgust': '🤢',
            'neutral': '😐'
        }
        
        self.setup_gui()
        self.initialize_components()
        
    def setup_gui(self):
        """Setup the GUI layout"""
        # Main frame
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = tk.Label(
            main_frame,
            text="😊 Emoticon - Facial Expression Recognition",
            font=('Arial', 24, 'bold'),
            fg='white',
            bg='#2c3e50'
        )
        title_label.pack(pady=(0, 20))
        
        # Content frame
        content_frame = tk.Frame(main_frame, bg='#2c3e50')
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Camera feed
        left_panel = tk.Frame(content_frame, bg='#34495e', relief=tk.RAISED, bd=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Camera label
        camera_label = tk.Label(
            left_panel,
            text="Live Camera Feed",
            font=('Arial', 16, 'bold'),
            fg='white',
            bg='#34495e'
        )
        camera_label.pack(pady=10)
        
        # Video frame
        self.video_label = tk.Label(left_panel, bg='black', width=640, height=480)
        self.video_label.pack(pady=10)
        
        # Right panel - Emotion display
        right_panel = tk.Frame(content_frame, bg='#34495e', relief=tk.RAISED, bd=2)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        
        # Emotion display
        emotion_frame = tk.Frame(right_panel, bg='#34495e')
        emotion_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Current emotion
        self.emotion_icon_label = tk.Label(
            emotion_frame,
            text="😐",
            font=('Arial', 72),
            bg='#34495e',
            fg='white'
        )
        self.emotion_icon_label.pack(pady=20)
        
        self.emotion_text_label = tk.Label(
            emotion_frame,
            text="Neutral",
            font=('Arial', 24, 'bold'),
            bg='#34495e',
            fg='white'
        )
        self.emotion_text_label.pack(pady=10)
        
        # Confidence bar
        confidence_frame = tk.Frame(emotion_frame, bg='#34495e')
        confidence_frame.pack(fill=tk.X, pady=20)
        
        tk.Label(
            confidence_frame,
            text="Confidence:",
            font=('Arial', 12),
            bg='#34495e',
            fg='white'
        ).pack()
        
        self.confidence_bar = ttk.Progressbar(
            confidence_frame,
            length=300,
            mode='determinate',
            style='Custom.Horizontal.TProgressbar'
        )
        self.confidence_bar.pack(pady=5)
        
        self.confidence_label = tk.Label(
            confidence_frame,
            text="0%",
            font=('Arial', 12),
            bg='#34495e',
            fg='white'
        )
        self.confidence_label.pack()
        
        # Statistics
        stats_frame = tk.Frame(emotion_frame, bg='#34495e')
        stats_frame.pack(fill=tk.X, pady=20)
        
        # FPS
        fps_frame = tk.Frame(stats_frame, bg='#34495e')
        fps_frame.pack(fill=tk.X, pady=5)
        tk.Label(fps_frame, text="FPS:", font=('Arial', 12), bg='#34495e', fg='white').pack(side=tk.LEFT)
        self.fps_label = tk.Label(fps_frame, text="0", font=('Arial', 12), bg='#34495e', fg='#f39c12')
        self.fps_label.pack(side=tk.RIGHT)
        
        # Frame count
        frame_frame = tk.Frame(stats_frame, bg='#34495e')
        frame_frame.pack(fill=tk.X, pady=5)
        tk.Label(frame_frame, text="Frames:", font=('Arial', 12), bg='#34495e', fg='white').pack(side=tk.LEFT)
        self.frame_label = tk.Label(frame_frame, text="0", font=('Arial', 12), bg='#34495e', fg='#f39c12')
        self.frame_label.pack(side=tk.RIGHT)
        
        # Control buttons
        button_frame = tk.Frame(main_frame, bg='#2c3e50')
        button_frame.pack(fill=tk.X, pady=20)
        
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
        
        # Status bar
        self.status_label = tk.Label(
            main_frame,
            text="Ready to start",
            font=('Arial', 10),
            bg='#2c3e50',
            fg='#ecf0f1'
        )
        self.status_label.pack(side=tk.BOTTOM, pady=10)
        
        # Configure custom progress bar style
        style = ttk.Style()
        style.theme_use('default')
        style.configure(
            'Custom.Horizontal.TProgressbar',
            troughcolor='#34495e',
            background='#3498db',
            bordercolor='#34495e',
            lightcolor='#3498db',
            darkcolor='#3498db'
        )
        
    def initialize_components(self):
        """Initialize the emotion detection components"""
        try:
            config_dir = Path(__file__).parent.parent / "config"
            
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
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            messagebox.showerror("Error", f"Failed to initialize components: {e}")
            self.status_label.config(text="Failed to initialize components")
    
    def start_detection(self):
        """Start the emotion detection"""
        if not self.is_running:
            try:
                self.camera_manager.start()
                self.is_running = True
                self.start_button.config(state=tk.DISABLED)
                self.stop_button.config(state=tk.NORMAL)
                self.status_label.config(text="Detection started")
                
                # Start the detection thread
                self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
                self.detection_thread.start()
                
            except Exception as e:
                logger.error(f"Failed to start detection: {e}")
                messagebox.showerror("Error", f"Failed to start detection: {e}")
    
    def stop_detection(self):
        """Stop the emotion detection"""
        if self.is_running:
            self.is_running = False
            self.camera_manager.stop()
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.status_label.config(text="Detection stopped")
    
    def detection_loop(self):
        """Main detection loop"""
        frame_count = 0
        start_time = time.time()
        
        while self.is_running:
            try:
                # Get frame from camera
                frame = self.camera_manager.get_frame()
                if frame is None:
                    continue
                
                frame_count += 1
                
                # Detect faces
                faces = self.face_detector.detect_faces(frame)
                
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
                
            except Exception as e:
                logger.error(f"Error in detection loop: {e}")
                continue
    
    def update_gui(self, frame, emotions, frame_count, start_time):
        """Update the GUI with detection results"""
        try:
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
                self.video_label.configure(image=photo)
                self.video_label.image = photo  # Keep a reference
                
                # Update emotion display
                if emotions:
                    emotion = emotions[0]  # Show first detected emotion
                    self.current_emotion = emotion['emotion']
                    self.current_confidence = emotion['confidence']
                    
                    # Update emotion icon and text
                    self.emotion_icon_label.config(text=self.emotion_icons.get(self.current_emotion, "😐"))
                    self.emotion_text_label.config(
                        text=self.current_emotion.title(),
                        fg=self.emotion_colors.get(self.current_emotion, "white")
                    )
                    
                    # Update confidence bar
                    confidence_percent = self.current_confidence * 100
                    self.confidence_bar['value'] = confidence_percent
                    self.confidence_label.config(text=f"{confidence_percent:.1f}%")
                
                # Update statistics
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                
                self.fps_label.config(text=f"{fps:.1f}")
                self.frame_label.config(text=str(frame_count))
                
        except Exception as e:
            logger.error(f"Error updating GUI: {e}")
    
    def on_closing(self):
        """Handle application closing"""
        if self.is_running:
            self.stop_detection()
        self.root.destroy()

def main():
    """Main entry point"""
    root = tk.Tk()
    app = EmoticonGUI(root)
    
    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Start the GUI
    root.mainloop()

if __name__ == "__main__":
    main()
