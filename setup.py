#!/usr/bin/env python3
"""
Setup script for Emoticon
Facial Expression Recognition for NVIDIA Jetson
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
install_requires = []
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        install_requires = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="emoticon",
    version="1.0.0",
    author="Vipul Sindha",
    author_email="vipul.sindha@gmail.com",
    description="Real-time facial expression recognition for NVIDIA Jetson hardware",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vipul-sindha/Emoticon",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "emoticon=src.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    keywords=[
        "emotion",
        "facial-expression",
        "computer-vision",
        "deep-learning",
        "jetson",
        "nvidia",
        "opencv",
        "pytorch",
        "tensorflow",
    ],
    project_urls={
        "Bug Reports": "https://github.com/vipul-sindha/Emoticon/issues",
        "Source": "https://github.com/vipul-sindha/Emoticon",
        "Documentation": "https://github.com/vipul-sindha/Emoticon#readme",
    },
)
