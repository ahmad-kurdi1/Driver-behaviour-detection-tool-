"""
Helper functions for Driver Monitoring System using dlib for face analysis
"""

import cv2
import numpy as np
import time
from collections import deque
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Import configuration
try:
    from config import DLIB_CONFIG, CLASS_COLORS, DLIB_LANDMARK_PATH, load_class_names_from_yaml
    CLASS_NAMES = load_class_names_from_yaml()
except ImportError:
    DLIB_CONFIG = {
        'EAR_THRESHOLD': 0.20,
        'HEAD_YAW_THRESHOLD': 25,
        'DROWSY_FRAME_THRESHOLD': 15,
        'EAR_HISTORY_SIZE': 30
    }
    CLASS_COLORS = {i: (0, 255, 0) for i in range(10)}
    CLASS_NAMES = {i: f"Class {i}" for i in range(10)}
    DLIB_LANDMARK_PATH = Path(__file__).parent / 'shape_predictor_68_face_landmarks.dat'

# Import dlib
try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    print("WARNING: dlib not installed. Run: pip install dlib-bin")


class FaceAnalyzerDlib:
    """Face analysis class using dlib"""
    
    def __init__(self, landmark_path: Optional[Path] = None):
        """Initialize face analyzer with dlib"""
        self.available = False
        
        if not DLIB_AVAILABLE:
            print("ERROR: dlib not available")
            return
        
        if landmark_path is None:
            landmark_path = DLIB_LANDMARK_PATH
        
        if not Path(landmark_path).exists():
            print(f"ERROR: Landmark file not found: {landmark_path}")
            print("   Download from:")
            print("   https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat")
            return
        
        try:
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(str(landmark_path))
            
            self.LEFT_EYE_INDICES = list(range(36, 42))
            self.RIGHT_EYE_INDICES = list(range(42, 48))
            
            self.ear_history = deque(maxlen=DLIB_CONFIG['EAR_HISTORY_SIZE'])
            self.drowsy_frames = 0
            
            self.available = True
            print("FaceAnalyzer dlib ready")
            
        except Exception as e:
            print(f"ERROR: Failed to initialize dlib: {e}")
            self.available = False
    
    def calculate_ear(self, landmarks, eye_indices: List[int]) -> float:
        """Calculate Eye Aspect Ratio (EAR)"""
        points = []
        for idx in eye_indices:
            point = (landmarks.part(idx).x, landmarks.part(idx).y)
            points.append(np.array(point))
        
        v1 = np.linalg.norm(points[1] - points[5])
        v2 = np.linalg.norm(points[2] - points[4])
        h = np.linalg.norm(points[0] - points[3])
        
        if h == 0:
            return 0
        return (v1 + v2) / (2.0 * h)
    
    def calculate_head_pose(self, landmarks) -> Tuple[float, str]:
        """Calculate head pose (approximate yaw angle)"""
        nose_tip = np.array([landmarks.part(30).x, landmarks.part(30).y])
        left_eye = np.array([landmarks.part(36).x, landmarks.part(36).y])
        right_eye = np.array([landmarks.part(45).x, landmarks.part(45).y])
        
        eye_center = (left_eye + right_eye) / 2
        nose_offset = nose_tip[0] - eye_center[0]
        eye_distance = np.linalg.norm(left_eye - right_eye)
        
        if eye_distance == 0:
            return 0, "Forward"
        
        yaw_ratio = nose_offset / eye_distance
        yaw_angle = yaw_ratio * 45
        
        if abs(yaw_angle) > DLIB_CONFIG['HEAD_YAW_THRESHOLD']:
            direction = "Left" if yaw_angle < 0 else "Right"
        else:
            direction = "Forward"
        
        return yaw_angle, direction
    
    def analyze(self, frame: np.ndarray) -> Dict:
        """Analyze a single frame and extract face information"""
        result = {
            'face_detected': False,
            'landmarks': None,
            'ear_left': 1.0,
            'ear_right': 1.0,
            'ear_avg': 1.0,
            'yaw_angle': 0,
            'direction': 'N/A',
            'drowsy': False,
            'face_rect': None
        }
        
        if not self.available:
            return result
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)
            
            if len(faces) > 0:
                face = faces[0]
                result['face_detected'] = True
                result['face_rect'] = (face.left(), face.top(), face.right(), face.bottom())
                
                landmarks = self.predictor(gray, face)
                result['landmarks'] = landmarks
                
                ear_left = self.calculate_ear(landmarks, self.LEFT_EYE_INDICES)
                ear_right = self.calculate_ear(landmarks, self.RIGHT_EYE_INDICES)
                ear_avg = (ear_left + ear_right) / 2.0
                
                result['ear_left'] = ear_left
                result['ear_right'] = ear_right
                result['ear_avg'] = ear_avg
                
                self.ear_history.append(ear_avg)
                smooth_ear = np.mean(list(self.ear_history)) if self.ear_history else ear_avg
                
                if smooth_ear < DLIB_CONFIG['EAR_THRESHOLD']:
                    self.drowsy_frames += 1
                else:
                    self.drowsy_frames = max(0, self.drowsy_frames - 1)
                
                result['drowsy'] = self.drowsy_frames >= DLIB_CONFIG['DROWSY_FRAME_THRESHOLD']
                
                yaw_angle, direction = self.calculate_head_pose(landmarks)
                result['yaw_angle'] = yaw_angle
                result['direction'] = direction
                
        except Exception as e:
            pass
        
        return result
    
    def draw_landmarks(self, frame: np.ndarray, landmarks) -> np.ndarray:
        """Draw facial landmarks on frame"""
        if landmarks is None:
            return frame
        
        try:
            for i in range(68):
                x = landmarks.part(i).x
                y = landmarks.part(i).y
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            
            for i in self.LEFT_EYE_INDICES + self.RIGHT_EYE_INDICES:
                x = landmarks.part(i).x
                y = landmarks.part(i).y
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
                
        except Exception:
            pass
        
        return frame


class AlertSystem:
    """Audio alert system"""
    
    def __init__(self, cooldown: float = 3.0):
        self.cooldown = cooldown
        self.last_alert_time = 0
        self.alert_count = 0
        
        self.sound_enabled = False
        try:
            import pygame
            pygame.mixer.init()
            self.sound_enabled = True
        except ImportError:
            print("WARNING: pygame not installed - audio alerts disabled")
    
    def play_alert(self, alert_type: str = "distraction"):
        """Play audio alert"""
        current_time = time.time()
        if current_time - self.last_alert_time < self.cooldown:
            return
        
        if self.sound_enabled:
            try:
                import pygame
                
                freq = 880 if alert_type == "drowsy" else 440
                duration = 0.3
                sample_rate = 44100
                
                t = np.linspace(0, duration, int(sample_rate * duration))
                wave = np.sin(2 * np.pi * freq * t)
                wave = (wave * 32767).astype(np.int16)
                stereo_wave = np.column_stack((wave, wave))
                
                sound = pygame.sndarray.make_sound(stereo_wave)
                sound.play()
            except Exception:
                pass
        
        self.last_alert_time = current_time
        self.alert_count += 1


def draw_results(frame: np.ndarray,
                 yolo_results: List[Dict],
                 face_data: Dict,
                 stats: Dict,
                 fps: float,
                 show_landmarks: bool = False,
                 face_analyzer: Optional[FaceAnalyzerDlib] = None) -> np.ndarray:
    """Draw results on frame"""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    cv2.rectangle(overlay, (0, 0), (w, 110), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
    
    y_pos = 30
    
    # Main status
    if face_data['drowsy']:
        status_text = "WARNING: Drowsiness Detected!"
        status_color = (0, 0, 255)
    elif abs(face_data['yaw_angle']) > DLIB_CONFIG['HEAD_YAW_THRESHOLD']:
        status_text = f"WARNING: Head Turned {face_data['direction']}"
        status_color = (0, 165, 255)
    elif yolo_results:
        main_obj = yolo_results[0]
        status_text = f"WARNING: {main_obj['class_name']} ({main_obj['confidence']:.2f})"
        status_color = (0, 255, 255)
    else:
        status_text = "Safe Driving"
        status_color = (0, 255, 0)
    
    cv2.putText(frame, status_text, (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    y_pos += 25
    ear_value = face_data['ear_avg']
    ear_text = f"EAR: {ear_value:.3f}"
    ear_color = (0, 0, 255) if ear_value < DLIB_CONFIG['EAR_THRESHOLD'] else (255, 255, 255)
    cv2.putText(frame, ear_text, (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, ear_color, 1)
    
    y_pos += 20
    direction = face_data['direction']
    yaw = face_data['yaw_angle']
    head_text = f"Head: {direction} ({abs(yaw):.1f} deg)"
    head_color = (0, 0, 255) if abs(yaw) > DLIB_CONFIG['HEAD_YAW_THRESHOLD'] else (255, 255, 255)
    cv2.putText(frame, head_text, (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, head_color, 1)
    
    y_pos += 20
    face_status = "Face Detected" if face_data['face_detected'] else "No Face"
    face_color = (0, 255, 0) if face_data['face_detected'] else (0, 0, 255)
    cv2.putText(frame, face_status, (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_color, 1)
    
    # YOLO bounding boxes
    if yolo_results:
        for obj in yolo_results:
            x1, y1, x2, y2 = obj['bbox']
            class_id = obj['class_id']
            conf = obj['confidence']
            class_name = obj['class_name']
            
            color = CLASS_COLORS.get(class_id, (255, 255, 255))
            if class_id not in CLASS_COLORS:
                CLASS_COLORS[class_id] = (
                    (class_id * 50) % 255,
                    (class_id * 80) % 255,
                    (class_id * 110) % 255
                )
                color = CLASS_COLORS[class_id]
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"{class_name}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # dlib face rectangle
    if face_data['face_detected'] and face_data['face_rect']:
        x1, y1, x2, y2 = face_data['face_rect']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 1)
    
    # Facial landmarks
    if show_landmarks and face_analyzer and face_data['landmarks']:
        face_analyzer.draw_landmarks(frame, face_data['landmarks'])
    
    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 100, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Statistics
    total = max(stats['total'], 1)
    safe_pct = stats['safe'] / total * 100
    stats_text = f"Safe: {safe_pct:.0f}% | Frames: {total}"
    cv2.putText(frame, stats_text, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    return frame


def format_time(seconds: float) -> str:
    """Format time for display"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
