#!/usr/bin/env python3
"""
Debug version of the GUI app with logging enabled
"""

import logging
import sys
import os

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s:%(name)s:%(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('gui_debug.log')
    ]
)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the GUI
if __name__ == "__main__":
    print("Starting GUI with debug logging...")
    exec(open('src/gui_app.py').read())
