#!/usr/bin/env python3
"""
Web Server for Emoticon
Provides REST API and web interface for facial expression recognition
"""

import json
import time
import logging
import threading
from flask import Flask, render_template, jsonify, request, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
from typing import Dict, Any, Optional
import base64
from io import BytesIO
from PIL import Image
from pathlib import Path

logger = logging.getLogger(__name__)

class WebServer:
    """Web server for emotion recognition system"""
    
    def __init__(self, host='0.0.0.0', port=8080):
        """Initialize web server"""
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        CORS(self.app)
        
        # Current results storage
        self.current_results = {
            'emotions': [],
            'timestamp': None,
            'frame_count': 0,
            'fps': 0.0
        }
        
        # Emotion history
        self.emotion_history = []
        self.max_history = 100
        
        # Setup routes
        self._setup_routes()
        
        # Server thread
        self.server_thread = None
        self.running = False
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            """Main page"""
            return render_template('index.html')
        
        @self.app.route('/api/emotion')
        def get_current_emotion():
            """Get current emotion detection results"""
            return jsonify(self.current_results)
        
        @self.app.route('/api/emotions/history')
        def get_emotion_history():
            """Get emotion detection history"""
            return jsonify({
                'history': self.emotion_history,
                'count': len(self.emotion_history)
            })
        
        @self.app.route('/api/status')
        def get_status():
            """Get system status"""
            return jsonify({
                'status': 'running' if self.running else 'stopped',
                'timestamp': time.time(),
                'frame_count': self.current_results['frame_count'],
                'fps': self.current_results['fps']
            })
        
        @self.app.route('/api/config')
        def get_config():
            """Get current configuration"""
            return jsonify({
                'host': self.host,
                'port': self.port,
                'max_history': self.max_history
            })
        
        @self.app.route('/api/emotions/stats')
        def get_emotion_stats():
            """Get emotion statistics"""
            if not self.emotion_history:
                return jsonify({'stats': {}})
            
            # Count emotions
            emotion_counts = {}
            for entry in self.emotion_history:
                emotion = entry.get('emotion', 'unknown')
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            # Calculate percentages
            total = len(self.emotion_history)
            emotion_percentages = {
                emotion: (count / total) * 100 
                for emotion, count in emotion_counts.items()
            }
            
            return jsonify({
                'stats': {
                    'counts': emotion_counts,
                    'percentages': emotion_percentages,
                    'total_detections': total
                }
            })
        
        @self.app.route('/api/clear_history', methods=['POST'])
        def clear_history():
            """Clear emotion history"""
            self.emotion_history.clear()
            return jsonify({'status': 'success', 'message': 'History cleared'})
        
        @self.app.route('/api/frame')
        def get_current_frame():
            """Get current frame as base64 image"""
            if 'frame' not in self.current_results or self.current_results['frame'] is None:
                return jsonify({'error': 'No frame available'})
            
            try:
                # Encode frame as base64
                frame = self.current_results['frame']
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                return jsonify({
                    'frame': frame_base64,
                    'timestamp': self.current_results['timestamp']
                })
            except Exception as e:
                logger.error(f"Error encoding frame: {e}")
                return jsonify({'error': 'Failed to encode frame'})
        
        # WebSocket events
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            logger.info("Client connected to WebSocket")
            emit('status', {'status': 'connected'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            logger.info("Client disconnected from WebSocket")
        
        @self.socketio.on('request_update')
        def handle_request_update():
            """Handle update request from client"""
            emit('emotion_update', self.current_results)
    
    def start(self):
        """Start the web server"""
        try:
            self.running = True
            self.server_thread = threading.Thread(
                target=self._run_server,
                daemon=True
            )
            self.server_thread.start()
            logger.info(f"Web server started on {self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"Failed to start web server: {e}")
            raise
    
    def _run_server(self):
        """Run the Flask server"""
        try:
            self.socketio.run(
                self.app,
                host=self.host,
                port=self.port,
                debug=False,
                use_reloader=False
            )
        except Exception as e:
            logger.error(f"Web server error: {e}")
    
    def stop(self):
        """Stop the web server"""
        self.running = False
        logger.info("Web server stopped")
    
    def update_results(self, results: Dict[str, Any]):
        """Update current results and broadcast to clients"""
        try:
            # Update current results
            self.current_results.update(results)
            self.current_results['timestamp'] = time.time()
            
            # Add to history if emotions detected
            if results.get('emotions'):
                for emotion_data in results['emotions']:
                    history_entry = {
                        'emotion': emotion_data['emotion'],
                        'confidence': emotion_data['confidence'],
                        'timestamp': time.time(),
                        'bbox': emotion_data['bbox']
                    }
                    self.emotion_history.append(history_entry)
                    
                    # Limit history size
                    if len(self.emotion_history) > self.max_history:
                        self.emotion_history.pop(0)
            
            # Broadcast to WebSocket clients
            self.socketio.emit('emotion_update', self.current_results)
            
        except Exception as e:
            logger.error(f"Error updating results: {e}")
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get server information"""
        return {
            'host': self.host,
            'port': self.port,
            'running': self.running,
            'clients_connected': len(self.socketio.server.manager.rooms.get('/', {})) if self.socketio.server.manager else 0
        }

# Create HTML template
def create_html_template():
    """Create the HTML template for the web interface"""
    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Emoticon - Facial Expression Recognition</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        .dashboard {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }
        .card {
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        .card h3 {
            margin-top: 0;
            color: #fff;
            border-bottom: 2px solid rgba(255,255,255,0.3);
            padding-bottom: 10px;
        }
        .emotion-display {
            font-size: 2em;
            font-weight: bold;
            text-align: center;
            margin: 20px 0;
            padding: 20px;
            border-radius: 10px;
            background: rgba(255,255,255,0.1);
        }
        .confidence-bar {
            width: 100%;
            height: 20px;
            background: rgba(255,255,255,0.2);
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #4CAF50, #8BC34A);
            transition: width 0.3s ease;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }
        .stat-item {
            text-align: center;
            padding: 10px;
            background: rgba(255,255,255,0.1);
            border-radius: 8px;
        }
        .stat-value {
            font-size: 1.5em;
            font-weight: bold;
        }
        .stat-label {
            font-size: 0.9em;
            opacity: 0.8;
        }
        .video-container {
            text-align: center;
            margin: 20px 0;
        }
        #videoCanvas {
            border-radius: 10px;
            max-width: 100%;
            height: auto;
        }
        .controls {
            text-align: center;
            margin: 20px 0;
        }
        .btn {
            background: rgba(255,255,255,0.2);
            border: none;
            color: white;
            padding: 10px 20px;
            border-radius: 25px;
            cursor: pointer;
            margin: 0 10px;
            transition: all 0.3s ease;
        }
        .btn:hover {
            background: rgba(255,255,255,0.3);
            transform: translateY(-2px);
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 10px;
        }
        .status-connected {
            background: #4CAF50;
        }
        .status-disconnected {
            background: #f44336;
        }
        @media (max-width: 768px) {
            .dashboard {
                grid-template-columns: 1fr;
            }
            .header h1 {
                font-size: 2em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>😊 Emoticon</h1>
            <p>Real-time Facial Expression Recognition</p>
            <div>
                <span class="status-indicator" id="statusIndicator"></span>
                <span id="statusText">Connecting...</span>
            </div>
        </div>
        
        <div class="dashboard">
            <div class="card">
                <h3>Current Emotion</h3>
                <div class="emotion-display" id="currentEmotion">Waiting...</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" id="confidenceBar" style="width: 0%"></div>
                </div>
                <div id="confidenceText">Confidence: 0%</div>
            </div>
            
            <div class="card">
                <h3>System Status</h3>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-value" id="fpsValue">0</div>
                        <div class="stat-label">FPS</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="frameCount">0</div>
                        <div class="stat-label">Frames</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="detectionCount">0</div>
                        <div class="stat-label">Detections</div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h3>Live Video Feed</h3>
            <div class="video-container">
                <canvas id="videoCanvas" width="640" height="480"></canvas>
            </div>
        </div>
        
        <div class="controls">
            <button class="btn" onclick="clearHistory()">Clear History</button>
            <button class="btn" onclick="refreshStats()">Refresh Stats</button>
        </div>
    </div>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script>
        const socket = io();
        const canvas = document.getElementById('videoCanvas');
        const ctx = canvas.getContext('2d');
        
        // Status management
        function updateStatus(connected) {
            const indicator = document.getElementById('statusIndicator');
            const text = document.getElementById('statusText');
            
            if (connected) {
                indicator.className = 'status-indicator status-connected';
                text.textContent = 'Connected';
            } else {
                indicator.className = 'status-indicator status-disconnected';
                text.textContent = 'Disconnected';
            }
        }
        
        // Socket event handlers
        socket.on('connect', function() {
            updateStatus(true);
            console.log('Connected to server');
        });
        
        socket.on('disconnect', function() {
            updateStatus(false);
            console.log('Disconnected from server');
        });
        
        socket.on('emotion_update', function(data) {
            updateEmotionDisplay(data);
            updateStats(data);
        });
        
        // Update emotion display
        function updateEmotionDisplay(data) {
            const emotionDisplay = document.getElementById('currentEmotion');
            const confidenceBar = document.getElementById('confidenceBar');
            const confidenceText = document.getElementById('confidenceText');
            
            if (data.emotions && data.emotions.length > 0) {
                const emotion = data.emotions[0];
                emotionDisplay.textContent = emotion.emotion.toUpperCase();
                const confidence = Math.round(emotion.confidence * 100);
                confidenceBar.style.width = confidence + '%';
                confidenceText.textContent = `Confidence: ${confidence}%`;
            } else {
                emotionDisplay.textContent = 'No Face Detected';
                confidenceBar.style.width = '0%';
                confidenceText.textContent = 'Confidence: 0%';
            }
        }
        
        // Update statistics
        function updateStats(data) {
            document.getElementById('fpsValue').textContent = data.fps ? data.fps.toFixed(1) : '0';
            document.getElementById('frameCount').textContent = data.frame_count || '0';
            document.getElementById('detectionCount').textContent = data.emotions ? data.emotions.length : '0';
        }
        
        // API functions
        function clearHistory() {
            fetch('/api/clear_history', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        console.log('History cleared');
                    }
                })
                .catch(error => console.error('Error clearing history:', error));
        }
        
        function refreshStats() {
            fetch('/api/emotions/stats')
                .then(response => response.json())
                .then(data => {
                    console.log('Stats updated:', data);
                })
                .catch(error => console.error('Error refreshing stats:', error));
        }
        
        // Request periodic updates
        setInterval(() => {
            socket.emit('request_update');
        }, 1000);
        
        // Initial status
        updateStatus(false);
    </script>
</body>
</html>
"""
    
    # Create templates directory
    templates_dir = Path(__file__).parent.parent / 'templates'
    templates_dir.mkdir(exist_ok=True)
    
    # Write template file
    template_file = templates_dir / 'index.html'
    with open(template_file, 'w') as f:
        f.write(html_template)
    
    logger.info(f"Created HTML template at {template_file}")

# Create template when module is imported
create_html_template()
