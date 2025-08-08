#!/usr/bin/env python3
"""
Visualization Utilities for Emoticon
Handles drawing results on frames and creating charts
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import List, Dict, Any, Tuple, Optional
import json
import time

logger = logging.getLogger(__name__)

class Visualizer:
    """Visualization utilities for emotion recognition results"""
    
    def __init__(self, enabled: bool = True):
        """Initialize visualizer"""
        self.enabled = enabled
        self.emotion_colors = {
            'happy': (0, 255, 0),      # Green
            'sad': (255, 0, 0),        # Blue
            'angry': (0, 0, 255),      # Red
            'fear': (128, 0, 128),     # Purple
            'surprise': (0, 255, 255), # Yellow
            'disgust': (0, 128, 0),    # Dark Green
            'neutral': (128, 128, 128) # Gray
        }
        
        # Set matplotlib style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def is_enabled(self) -> bool:
        """Check if visualization is enabled"""
        return self.enabled
    
    def draw_results(self, frame: np.ndarray, emotions: List[Dict[str, Any]]) -> np.ndarray:
        """Draw emotion detection results on frame"""
        if not self.enabled or frame is None:
            return frame
        
        try:
            annotated_frame = frame.copy()
            
            for emotion_data in emotions:
                bbox = emotion_data.get('bbox')
                emotion = emotion_data.get('emotion', 'unknown')
                confidence = emotion_data.get('confidence', 0.0)
                
                if bbox is not None:
                    # Draw bounding box
                    x, y, w, h = bbox
                    color = self.emotion_colors.get(emotion, (255, 255, 255))
                    
                    cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
                    
                    # Draw emotion label
                    label = f"{emotion.upper()}: {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    
                    # Draw label background
                    cv2.rectangle(annotated_frame, 
                                (x, y - label_size[1] - 10), 
                                (x + label_size[0], y), 
                                color, -1)
                    
                    # Draw label text
                    cv2.putText(annotated_frame, label, 
                              (x, y - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            return annotated_frame
            
        except Exception as e:
            logger.error(f"Error drawing results: {e}")
            return frame
    
    def draw_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """Draw FPS counter on frame"""
        if not self.enabled or frame is None:
            return frame
        
        try:
            annotated_frame = frame.copy()
            
            # Draw FPS text
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(annotated_frame, fps_text, 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            return annotated_frame
            
        except Exception as e:
            logger.error(f"Error drawing FPS: {e}")
            return frame
    
    def draw_timestamp(self, frame: np.ndarray, timestamp: float) -> np.ndarray:
        """Draw timestamp on frame"""
        if not self.enabled or frame is None:
            return frame
        
        try:
            annotated_frame = frame.copy()
            
            # Convert timestamp to readable format
            time_str = time.strftime("%H:%M:%S", time.localtime(timestamp))
            
            # Draw timestamp text
            cv2.putText(annotated_frame, time_str, 
                      (10, frame.shape[0] - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            return annotated_frame
            
        except Exception as e:
            logger.error(f"Error drawing timestamp: {e}")
            return frame
    
    def create_emotion_chart(self, emotion_history: List[Dict[str, Any]], 
                           save_path: Optional[str] = None) -> np.ndarray:
        """Create emotion distribution chart"""
        try:
            if not emotion_history:
                return np.array([])
            
            # Count emotions
            emotion_counts = {}
            for entry in emotion_history:
                emotion = entry.get('emotion', 'unknown')
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            if not emotion_counts:
                return np.array([])
            
            # Create pie chart
            fig, ax = plt.subplots(figsize=(10, 8))
            
            emotions = list(emotion_counts.keys())
            counts = list(emotion_counts.values())
            colors = [self.emotion_colors.get(emotion, (0.5, 0.5, 0.5)) for emotion in emotions]
            
            # Convert BGR to RGB for matplotlib
            colors_rgb = [(c[2]/255, c[1]/255, c[0]/255) for c in colors]
            
            wedges, texts, autotexts = ax.pie(counts, labels=emotions, colors=colors_rgb, 
                                             autopct='%1.1f%%', startangle=90)
            
            ax.set_title('Emotion Distribution', fontsize=16, fontweight='bold')
            
            # Convert plot to image
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            plt.close(fig)
            
            # Save if path provided
            if save_path:
                cv2.imwrite(save_path, img_bgr)
                logger.info(f"Emotion chart saved to {save_path}")
            
            return img_bgr
            
        except Exception as e:
            logger.error(f"Error creating emotion chart: {e}")
            return np.array([])
    
    def create_confidence_plot(self, emotion_history: List[Dict[str, Any]], 
                             save_path: Optional[str] = None) -> np.ndarray:
        """Create confidence over time plot"""
        try:
            if not emotion_history:
                return np.array([])
            
            # Extract timestamps and confidences
            timestamps = []
            confidences = []
            emotions = []
            
            for entry in emotion_history:
                timestamp = entry.get('timestamp', 0)
                confidence = entry.get('confidence', 0.0)
                emotion = entry.get('emotion', 'unknown')
                
                timestamps.append(timestamp)
                confidences.append(confidence)
                emotions.append(emotion)
            
            if not timestamps:
                return np.array([])
            
            # Create line plot
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot confidence over time
            ax.plot(timestamps, confidences, 'b-', alpha=0.7, linewidth=2)
            ax.scatter(timestamps, confidences, c='red', s=30, alpha=0.8)
            
            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel('Confidence', fontsize=12)
            ax.set_title('Emotion Detection Confidence Over Time', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Set y-axis limits
            ax.set_ylim(0, 1)
            
            # Convert plot to image
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            plt.close(fig)
            
            # Save if path provided
            if save_path:
                cv2.imwrite(save_path, img_bgr)
                logger.info(f"Confidence plot saved to {save_path}")
            
            return img_bgr
            
        except Exception as e:
            logger.error(f"Error creating confidence plot: {e}")
            return np.array([])
    
    def create_emotion_timeline(self, emotion_history: List[Dict[str, Any]], 
                              save_path: Optional[str] = None) -> np.ndarray:
        """Create emotion timeline visualization"""
        try:
            if not emotion_history:
                return np.array([])
            
            # Extract data
            timestamps = []
            emotions = []
            
            for entry in emotion_history:
                timestamp = entry.get('timestamp', 0)
                emotion = entry.get('emotion', 'unknown')
                
                timestamps.append(timestamp)
                emotions.append(emotion)
            
            if not timestamps:
                return np.array([])
            
            # Create timeline plot
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # Create emotion mapping
            unique_emotions = list(set(emotions))
            emotion_to_y = {emotion: i for i, emotion in enumerate(unique_emotions)}
            
            # Plot timeline
            y_positions = [emotion_to_y[emotion] for emotion in emotions]
            
            for i, (timestamp, emotion) in enumerate(zip(timestamps, emotions)):
                color = self.emotion_colors.get(emotion, (0.5, 0.5, 0.5))
                color_rgb = (color[2]/255, color[1]/255, color[0]/255)
                
                ax.scatter(timestamp, emotion_to_y[emotion], 
                          c=[color_rgb], s=100, alpha=0.8, label=emotion if i == 0 else "")
            
            ax.set_yticks(range(len(unique_emotions)))
            ax.set_yticklabels(unique_emotions)
            ax.set_xlabel('Time', fontsize=12)
            ax.set_title('Emotion Timeline', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Remove duplicate legend entries
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
            
            # Convert plot to image
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            plt.close(fig)
            
            # Save if path provided
            if save_path:
                cv2.imwrite(save_path, img_bgr)
                logger.info(f"Emotion timeline saved to {save_path}")
            
            return img_bgr
            
        except Exception as e:
            logger.error(f"Error creating emotion timeline: {e}")
            return np.array([])
    
    def create_summary_dashboard(self, emotion_history: List[Dict[str, Any]], 
                               save_path: Optional[str] = None) -> np.ndarray:
        """Create a comprehensive dashboard with multiple visualizations"""
        try:
            if not emotion_history:
                return np.array([])
            
            # Create subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. Emotion distribution pie chart
            emotion_counts = {}
            for entry in emotion_history:
                emotion = entry.get('emotion', 'unknown')
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            if emotion_counts:
                emotions = list(emotion_counts.keys())
                counts = list(emotion_counts.values())
                colors = [self.emotion_colors.get(emotion, (0.5, 0.5, 0.5)) for emotion in emotions]
                colors_rgb = [(c[2]/255, c[1]/255, c[0]/255) for c in colors]
                
                ax1.pie(counts, labels=emotions, colors=colors_rgb, autopct='%1.1f%%')
                ax1.set_title('Emotion Distribution', fontweight='bold')
            
            # 2. Confidence over time
            timestamps = [entry.get('timestamp', 0) for entry in emotion_history]
            confidences = [entry.get('confidence', 0.0) for entry in emotion_history]
            
            if timestamps and confidences:
                ax2.plot(timestamps, confidences, 'b-', alpha=0.7)
                ax2.scatter(timestamps, confidences, c='red', s=20, alpha=0.8)
                ax2.set_title('Confidence Over Time', fontweight='bold')
                ax2.set_ylabel('Confidence')
                ax2.grid(True, alpha=0.3)
            
            # 3. Emotion frequency bar chart
            if emotion_counts:
                emotions = list(emotion_counts.keys())
                counts = list(emotion_counts.values())
                colors = [self.emotion_colors.get(emotion, (0.5, 0.5, 0.5)) for emotion in emotions]
                colors_rgb = [(c[2]/255, c[1]/255, c[0]/255) for c in colors]
                
                bars = ax3.bar(emotions, counts, color=colors_rgb, alpha=0.8)
                ax3.set_title('Emotion Frequency', fontweight='bold')
                ax3.set_ylabel('Count')
                
                # Add value labels on bars
                for bar, count in zip(bars, counts):
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                            str(count), ha='center', va='bottom')
            
            # 4. Average confidence by emotion
            emotion_confidence = {}
            for entry in emotion_history:
                emotion = entry.get('emotion', 'unknown')
                confidence = entry.get('confidence', 0.0)
                
                if emotion not in emotion_confidence:
                    emotion_confidence[emotion] = []
                emotion_confidence[emotion].append(confidence)
            
            if emotion_confidence:
                avg_confidences = {emotion: np.mean(confidences) 
                                 for emotion, confidences in emotion_confidence.items()}
                
                emotions = list(avg_confidences.keys())
                avg_conf = list(avg_confidences.values())
                colors = [self.emotion_colors.get(emotion, (0.5, 0.5, 0.5)) for emotion in emotions]
                colors_rgb = [(c[2]/255, c[1]/255, c[0]/255) for c in colors]
                
                bars = ax4.bar(emotions, avg_conf, color=colors_rgb, alpha=0.8)
                ax4.set_title('Average Confidence by Emotion', fontweight='bold')
                ax4.set_ylabel('Average Confidence')
                ax4.set_ylim(0, 1)
                
                # Add value labels on bars
                for bar, conf in zip(bars, avg_conf):
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{conf:.2f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Convert plot to image
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            plt.close(fig)
            
            # Save if path provided
            if save_path:
                cv2.imwrite(save_path, img_bgr)
                logger.info(f"Summary dashboard saved to {save_path}")
            
            return img_bgr
            
        except Exception as e:
            logger.error(f"Error creating summary dashboard: {e}")
            return np.array([])
    
    def save_visualization_data(self, emotion_history: List[Dict[str, Any]], 
                               file_path: str):
        """Save visualization data to JSON file"""
        try:
            data = {
                'timestamp': time.time(),
                'total_detections': len(emotion_history),
                'emotion_history': emotion_history,
                'statistics': {
                    'emotion_counts': {},
                    'average_confidence': 0.0,
                    'detection_rate': 0.0
                }
            }
            
            # Calculate statistics
            if emotion_history:
                emotion_counts = {}
                confidences = []
                
                for entry in emotion_history:
                    emotion = entry.get('emotion', 'unknown')
                    confidence = entry.get('confidence', 0.0)
                    
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                    confidences.append(confidence)
                
                data['statistics']['emotion_counts'] = emotion_counts
                data['statistics']['average_confidence'] = np.mean(confidences)
                data['statistics']['detection_rate'] = len(emotion_history) / max(1, len(emotion_history))
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Visualization data saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving visualization data: {e}")
    
    def get_visualization_info(self) -> Dict[str, Any]:
        """Get visualization configuration information"""
        return {
            'enabled': self.enabled,
            'emotion_colors': self.emotion_colors,
            'available_charts': [
                'emotion_chart',
                'confidence_plot',
                'emotion_timeline',
                'summary_dashboard'
            ]
        }
