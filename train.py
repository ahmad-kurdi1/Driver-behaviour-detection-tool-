"""
Train YOLOv8 model on Roboflow Distracted Driver Dataset
"""

import sys
from pathlib import Path
import yaml

sys.path.append(str(Path(__file__).resolve().parent))

from ultralytics import YOLO
from config import DATASET_YAML, MODELS_DIR, YOLO_CONFIG


def check_dataset():
    """Check if dataset is properly configured"""
    print("\n" + "="*60)
    print("Checking Dataset Configuration")
    print("="*60)
    
    if not DATASET_YAML.exists():
        print(f"ERROR: data.yaml not found at: {DATASET_YAML}")
        return False
    
    print(f"Data yaml found: {DATASET_YAML}")
    
    try:
        with open(DATASET_YAML, 'r') as f:
            data = yaml.safe_load(f)
        
        print(f"Path: {data.get('path', 'N/A')}")
        print(f"Train: {data.get('train', 'N/A')}")
        print(f"Val: {data.get('val', 'N/A')}")
        print(f"Test: {data.get('test', 'N/A')}")
        print(f"Number of classes: {data.get('nc', 'N/A')}")
        print(f"Class names: {data.get('names', 'N/A')}")
        
        # Check if directories exist
        dataset_path = DATASET_YAML.parent
        train_path = dataset_path / data.get('train', 'train/images')
        val_path = dataset_path / data.get('val', 'valid/images')
        
        if train_path.exists():
            num_train = len(list(train_path.glob('*.jpg'))) + len(list(train_path.glob('*.png')))
            print(f"Training images: {num_train}")
        else:
            print(f"WARNING: Train path not found: {train_path}")
        
        if val_path.exists():
            num_val = len(list(val_path.glob('*.jpg'))) + len(list(val_path.glob('*.png')))
            print(f"Validation images: {num_val}")
        else:
            print(f"WARNING: Val path not found: {val_path}")
        
        return True
        
    except Exception as e:
        print(f"ERROR reading data.yaml: {e}")
        return False


def train_model():
    """Train YOLO model on the dataset"""
    print("\n" + "="*60)
    print("Training YOLOv8 on Distracted Driver Dataset")
    print("="*60)
    
    # Check dataset first
    if not check_dataset():
        print("\nPlease fix dataset issues before training.")
        return
    
    # Detect GPU
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load model
    print(f"\nLoading model: {YOLO_CONFIG['model_type']}")
    model = YOLO(YOLO_CONFIG['model_type'])
    
    # Training arguments
    train_args = {
        'data': str(DATASET_YAML),
        'epochs': 50,
        'patience': 20,
        'batch': 24,           # ✅ مناسب لـ 4GB VRAM
        'imgsz': 640,
        'device': device,
        'workers': 4,          # ✅ أقل لتجنب مشاكل الذاكرة
        'amp': True,           # ✅ يسرع بدون تأثير على الدقة
        'cache': 'disk',       # ✅ تخزين على القرص (أكثر استقراراً)
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'project': str(MODELS_DIR),
        'name': 'driver_behavior_roboflow',
        'exist_ok': True,
        'pretrained': True,
        'save': True,
        'save_period': 10,
        'plots': True,
        'val': True,
        'verbose': True,
    }
    
    print("\n" + "-"*40)
    print("Training Configuration:")
    print("-"*40)
    for key, value in train_args.items():
        print(f"  {key}: {value}")
    print("-"*40 + "\n")
    
    # Start training
    print("Starting training...")
    print("Press Ctrl+C to stop early (model will be saved)\n")
    
    try:
        results = model.train(**train_args)
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        print(f"Model saved to: {MODELS_DIR}/driver_behavior_roboflow/weights/best.pt")
        
        # Evaluate
        print("\nEvaluating model...")
        metrics = model.val()
        
        print("\nFinal Metrics:")
        if hasattr(metrics, 'box'):
            print(f"  mAP50: {metrics.box.map50:.4f}")
            print(f"  mAP50-95: {metrics.box.map:.4f}")
            print(f"  Precision: {metrics.box.mp:.4f}")
            print(f"  Recall: {metrics.box.mr:.4f}")
        
        print("\n" + "="*60)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Model weights have been saved.")


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train YOLO on Distracted Driver Dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=None, help='Batch size')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Base model')
    
    args = parser.parse_args()
    
    # Override config with arguments
    if args.model:
        YOLO_CONFIG['model_type'] = args.model
    
    train_model()


if __name__ == "__main__":
    main()