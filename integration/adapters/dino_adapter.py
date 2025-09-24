import sys
import os

# Add paths to both repositories
sys.path.append('/mnt/data/dino_cubify_integration/ml-cubifyanything')
sys.path.append('/mnt/data/dino_cubify_integration/MoGe')

import torch
import torch.nn as nn
import torch.nn.functional as F

def test_basic_integration():
    """Test basic functionality"""
    print("Testing basic integration...")
    
    # Test CUDA
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Test imports
    try:
        from moge.model.modules import DINOv2Encoder
        print("✓ MoGe import successful")
        
        from cubifyanything.batching import BatchedPosedSensor
        print("✓ CubifyAnything import successful")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

if __name__ == "__main__":
    test_basic_integration()
