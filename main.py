"""
============================================================================
Driver Behavior Monitoring System - YOLO11s + dlib
Maps COCO detections to Driver Behavior Classes
============================================================================
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import time
import argparse
from collections import deque

sys.path.append(str(Path(__file__).resolve().parent))

from ultralytics import YOLO
from config import (
    MODELS_DIR, CAMERA_CONFIG, DISPLAY_CONFIG, OUTPUT_DIR, DLIB_CONFIG
)
from utils.helpers import FaceAnalyzerDlib, AlertSystem, format_time


# ============================================================
# Map COCO classes to Driver Behavior Classes
# ============================================================
# COCO class IDs that are relevant to driver behavior
COCO_TO_DRIVER = {
    67: {  # cell phone
        'class_name': 'Talking on Phone / Texting',
        'color': (0, 140, 255),  # Orange
        'behavior': 'distracted'
    },
    41: {  # cup
        'class_name': 'Drinking',
        'color': (0, 0, 255),  # Red
        'behavior': 'distracted'
    },
    39: {  # bottle
        'class_name': 'Drinking',
        'color': (0, 0, 255),  # Red
        'behavior': 'distracted'
    },
    40: {  # wine glass
        'class_name': 'Drinking',
        'color': (0, 0, 255),  # Red
        'behavior': 'distracted'
    },
    45: {  # bowl
        'class_name': 'Eating / Drinking',
        'color': (0, 0, 255),  # Red
        'behavior': 'distracted'
    },
    46: {  # banana
        'class_name': 'Eating',
        'color': (255, 0, 255),  # Magenta
        'behavior': 'distracted'
    },
    52: {  # hot dog
        'class_name': 'Eating',
        'color': (255, 0, 255),  # Magenta
        'behavior': 'distracted'
    },
    53: {  # pizza
        'class_name': 'Eating',
        'color': (255, 0, 255),  # Magenta
        'behavior': 'distracted'
    },
    73: {  # book
        'class_name': 'Reading / Distracted',
        'color': (255, 255, 0),  # Cyan
        'behavior': 'distracted'
    },
    76: {  # scissors
        'class_name': 'Distracted (Object in Hand)',
        'color': (255, 255, 0),  # Cyan
        'behavior': 'distracted'
    },
}


def load_model_and_classes(model_path=None):
    """Load YOLO model"""
    if model_path is None:
        model_path = 'yolo11s.pt'
    
    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))
    
    # Use COCO class names
    class_names = model.names
    
    print(f"Model loaded: {len(class_names)} COCO classes")
    print("Mapped to Driver Behavior Classes:")
    for coco_id, behavior in COCO_TO_DRIVER.items():
        coco_name = class_names.get(coco_id, f"Class {coco_id}")
        print(f"  {coco_name} → {behavior['class_name']}")
    
    return model, class_names


def draw_results(frame, detections, face_data, stats, fps, show_landmarks=False, face_analyzer=None):
    """Draw detection results on frame"""
    h, w = frame.shape[:2]
    
    # Draw bounding boxes
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        color = det['color']
        cls_name = det['class_name']
        conf = det['confidence']
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name}: {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Draw face landmarks
    if show_landmarks and face_analyzer and face_data.get('landmarks'):
        face_analyzer.draw_landmarks(frame, face_data['landmarks'])
    
    # Draw face rectangle
    if face_data.get('face_rect'):
        x1, y1, x2, y2 = face_data['face_rect']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 1)
    
    # Status bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
    
    # Determine state
    if face_data['drowsy']:
        state = "⚠️ DROWSY!"
        state_color = (0, 0, 255)
    elif detections:
        state = f"⚠️ {detections[0]['class_name']}"
        state_color = (0, 255, 255)
    else:
        state = "✅ Safe Driving"
        state_color = (0, 255, 0)
    
    cv2.putText(frame, state, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
    
    # EAR and Head info
    ear = face_data.get('ear_avg', 1.0)
    direction = face_data.get('direction', 'N/A')
    cv2.putText(frame, f"EAR: {ear:.3f} | Head: {direction}", (10, 55),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Statistics
    total = max(stats['total'], 1)
    safe_pct = stats['safe'] / total * 100
    cv2.putText(frame, f"Safe: {safe_pct:.0f}% | Frames: {total}", (10, h - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 100, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    return frame


# ============================================================
# Main Function
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Driver Behavior Monitoring - YOLO11s + dlib')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--model', type=str, default='yolo11s.pt', help='YOLO model path')
    parser.add_argument('--show-landmarks', action='store_true', help='Show facial landmarks')
    parser.add_argument('--save', action='store_true', help='Save video output')
    parser.add_argument('--conf', type=float, default=0.4, help='Confidence threshold')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Driver Behavior Monitoring System - YOLO11s + dlib")
    print("="*60)
    
    # Load model
    model, COCO_CLASSES = load_model_and_classes(args.model)
    CONFIDENCE_THRESHOLD = args.conf
    
    # Initialize face analyzer
    print("\nInitializing dlib face analyzer...")
    face_analyzer = FaceAnalyzerDlib()
    
    # Alert system
    alert_system = AlertSystem()
    
    # Camera
    print(f"\nOpening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_CONFIG['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_CONFIG['height'])
    
    if not cap.isOpened():
        print(f"ERROR: Failed to open camera {args.camera}")
        return
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera: {w}x{h}")
    
    # Video writer
    writer = None
    if args.save:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"session_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, 30.0, (w, h))
        print(f"Saving video: {output_path}")
    
    print("\nControls:")
    print("  'q' = Quit | 'l' = Landmarks | 's' = Screenshot")
    print("="*60 + "\n")
    
    cv2.namedWindow(DISPLAY_CONFIG['window_name'], cv2.WINDOW_NORMAL)
    cv2.resizeWindow(DISPLAY_CONFIG['window_name'], 
                     DISPLAY_CONFIG['window_width'], 
                     DISPLAY_CONFIG['window_height'])
    
    show_landmarks = args.show_landmarks
    stats = {'total': 0, 'safe': 0, 'distracted': 0, 'drowsy': 0}
    start_time = time.time()
    prev_time = start_time
    frame_count = 0
    fps = 0
    consecutive_drowsy = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        stats['total'] += 1
        current_time = time.time()
        
        # ============================================================
        # YOLO Detection (COCO model)
        # ============================================================
        results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
        detections = []
        
        if results and results[0].boxes:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                # Map COCO class to driver behavior
                if cls_id in COCO_TO_DRIVER:
                    behavior = COCO_TO_DRIVER[cls_id]
                    detections.append({
                        'class_id': cls_id,
                        'coco_name': COCO_CLASSES.get(cls_id, f"Class {cls_id}"),
                        'class_name': behavior['class_name'],
                        'color': behavior['color'],
                        'behavior': behavior['behavior'],
                        'confidence': conf,
                        'bbox': (x1, y1, x2, y2)
                    })
        
        # ============================================================
        # Face Analysis (dlib)
        # ============================================================
        face_data = face_analyzer.analyze(frame)
        
        if face_data['ear_avg'] < DLIB_CONFIG['EAR_THRESHOLD']:
            consecutive_drowsy += 1
        else:
            consecutive_drowsy = max(0, consecutive_drowsy - 1)
        
        face_data['drowsy'] = consecutive_drowsy >= DLIB_CONFIG['DROWSY_FRAME_THRESHOLD']
        
        # ============================================================
        # Statistics & Alerts
        # ============================================================
        if face_data['drowsy']:
            stats['drowsy'] += 1
            alert_system.play_alert("drowsy")
        elif detections:
            stats['distracted'] += 1
            alert_system.play_alert("distraction")
        else:
            stats['safe'] += 1
        
        # ============================================================
        # FPS
        # ============================================================
        if current_time - prev_time >= 1.0:
            fps = frame_count / (current_time - start_time)
            prev_time = current_time
        
        # ============================================================
        # Draw Results
        # ============================================================
        display_frame = draw_results(
            frame, detections, face_data, stats, fps,
            show_landmarks, face_analyzer
        )
        
        # Time
        elapsed = current_time - start_time
        cv2.putText(display_frame, format_time(elapsed), (w - 150, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        if writer:
            writer.write(display_frame)
        
        cv2.imshow(DISPLAY_CONFIG['window_name'], display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('l'):
            show_landmarks = not show_landmarks
            print(f"Landmarks: {'ON' if show_landmarks else 'OFF'}")
        elif key == ord('s'):
            screenshot_path = OUTPUT_DIR / f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(str(screenshot_path), display_frame)
            print(f"Screenshot saved: {screenshot_path}")
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    # Summary
    print("\n" + "="*60)
    print("Session Summary")
    print("="*60)
    total = max(stats['total'], 1)
    elapsed = time.time() - start_time
    print(f"Duration: {format_time(elapsed)}")
    print(f"Total Frames: {stats['total']}")
    print(f"Safe Driving: {stats['safe']} ({100*stats['safe']/total:.1f}%)")
    print(f"Distracted: {stats['distracted']} ({100*stats['distracted']/total:.1f}%)")
    print(f"Drowsy: {stats['drowsy']} ({100*stats['drowsy']/total:.1f}%)")
    print(f"Alerts: {alert_system.alert_count}")
    print(f"Avg FPS: {stats['total']/elapsed:.1f}")
    print("="*60)


if __name__ == "__main__":
    main()