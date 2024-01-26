# Face Recognizer

## Introduction
Face Recognizer is a Python-based facial recognition system developed using OpenCV. This project aims to provide an efficient and easy-to-use tool for facial detection and recognition in real-time.

## Features
- **Real-Time Face Detection**: Detect faces in real-time using your webcam.
- **Training Data Preparation**: Easily prepare and manage your training data for facial recognition.
- **Face Recognition**: Recognize and identify faces from a database of known individuals.

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
- Python 3.x
- OpenCV library
- NumPy

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/SyedHassanUlHaq/Face-recognizer.git
   ```
2. Navigate to the cloned repository.
   ```bash
   cd Face-recognizer
   ```
3. Install the required packages:
   ```
   pip install opencv-contrib-python numpy
   ```
### Running the Application
To start the application, run:
```bash
python facemaster.py
```
### Additional Configuration
For some environments, particularly when running on Linux with certain display managers, you may need to set an environment variable for the Qt platform:
```bash
export QT_QPA_PLATFORM=xcb
```
Include this in your environment or run it in your terminal before starting the application if you encounter display-related issues.

### Usage

Place the images of the individuals you want to recognize in the `training-data` folder. Each individual's images should be in a separate subfolder. The application will use these images to train the face recognizer.