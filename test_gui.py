#!/usr/bin/env python3
"""
Simple GUI test to verify the window appears and can display images
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageTk
import cv2

def test_gui():
    """Test basic GUI functionality"""
    root = tk.Tk()
    root.title("GUI Test")
    root.geometry("800x600")
    
    # Create a test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(test_image, "GUI Test", (250, 240), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(test_image, "If you see this, GUI works!", (200, 280), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    # Convert to PIL and display
    test_image_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(test_image_rgb)
    photo = ImageTk.PhotoImage(pil_image)
    
    # Create label and display image
    label = tk.Label(root, image=photo)
    label.image = photo  # Keep a reference
    label.pack(pady=20)
    
    # Add a button
    button = tk.Button(root, text="Click me!", command=lambda: print("Button clicked!"))
    button.pack(pady=10)
    
    print("GUI test window should appear now...")
    root.mainloop()

if __name__ == "__main__":
    test_gui()
