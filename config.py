"""
Configuration file for Driver Behavior Monitoring System
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
OUTPUT_DIR = BASE_DIR / 'output'
UTILS_DIR = BASE_DIR / 'utils'

# Create directories if they don't exist
for dir_path in [MODELS_DIR, OUTPUT_DIR, DATA_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Dataset path - Updated for Roboflow dataset
DATASET_PATH = DATA_DIR / 'distracted-driver-detection-3'
DATASET_YAML = DATASET_PATH / 'data.yaml'

# dlib landmark file path
DLIB_LANDMARK_PATH = UTILS_DIR / 'shape_predictor_68_face_landmarks.dat'

# Class names - Will be loaded dynamically from data.yaml
# These are fallback values
CLASS_NAMES = {
    0: "drinking",
    1: "hair and makeup",
    2: "operating the radio",
    3: "reaching behind",
    4: "safe driving",
    5: "talking on the phone",
    6: "talking to passenger",
    7: "texting"
}

# Display colors for each class (BGR)
CLASS_COLORS = {
    0: (0, 255, 0),      # Green - Safe
    1: (0, 255, 255),    # Yellow - Texting Right
    2: (0, 165, 255),    # Orange - Phone Right
    3: (255, 255, 0),    # Cyan - Texting Left
    4: (255, 165, 0),    # Blue-Orange - Phone Left
    5: (255, 0, 255),    # Magenta - Radio
    6: (128, 0, 128),    # Purple - Drinking
    7: (128, 128, 0),    # Olive - Reaching
    8: (255, 192, 203),  # Pink - Makeup
    9: (0, 128, 255)     # Light Blue - Passenger
}

# dlib drowsiness detection settings
DLIB_CONFIG = {
    'EAR_THRESHOLD': 0.20,           # Eye Aspect Ratio threshold
    'HEAD_YAW_THRESHOLD': 25,        # Head turn angle threshold
    'DROWSY_FRAME_THRESHOLD': 15,    # Consecutive frames for drowsiness
    'EAR_HISTORY_SIZE': 30           # EAR history buffer size
}

# YOLO settings
YOLO_CONFIG = {
    'model_type': 'yolov8s.pt',
    'confidence_threshold': 0.3,
    'iou_threshold': 0.45,
    'device': 'cuda'  # Use GPU if available, else 'cpu'
}

# Alert settings
ALERT_CONFIG = {
    'alert_cooldown': 3.0,
    'drowsy_alert_sound': 880,
    'distraction_alert_sound': 440
}

# Camera settings
CAMERA_CONFIG = {
    'width': 640,
    'height': 480,
    'fps': 30
}

# Display settings
DISPLAY_CONFIG = {
    'window_name': 'Driver Behavior Monitoring System',
    'window_width': 900,
    'window_height': 700,
    'show_fps': True,
    'font_scale': 0.6,
    'font_thickness': 2
}


def load_class_names_from_yaml(yaml_path: Path = None) -> dict:
    """
    Load class names from data.yaml file
    
    Args:
        yaml_path: Path to data.yaml file
    
    Returns:
        Dictionary of class names
    """
    if yaml_path is None:
        yaml_path = DATASET_YAML
    
    if not yaml_path.exists():
        print(f"Warning: {yaml_path} not found. Using default class names.")
        return CLASS_NAMES
    
    try:
        import yaml
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        if 'names' in data:
            if isinstance(data['names'], list):
                return {i: name for i, name in enumerate(data['names'])}
            elif isinstance(data['names'], dict):
                return data['names']
        
        return CLASS_NAMES
    except Exception as e:
        print(f"Warning: Could not load class names from yaml: {e}")
        return CLASS_NAMES