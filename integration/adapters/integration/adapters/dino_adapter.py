import sys
import os

# Add paths to both repositories
sys.path.append('/mnt/data/dino_cubify_integration/ml-cubifyanything')
sys.path.append('/mnt/data/dino_cubify_integration/MoGe')

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union

def test_imports():
    """Test that all imports work"""
    try:
        print("Testing imports...")
        
        # Test CUDA
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"GPU count: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print(f"GPU 0: {torch.cuda.get_device_name(0)}")
        
        # Test MoGe
        from moge.model.modules import DINOv2Encoder
        print("✓ MoGe imports successful")
        
        # Test CubifyAnything
        from cubifyanything.batching import BatchedPosedSensor
        print("✓ CubifyAnything imports successful")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dino_encoder():
    """Test DINO encoder creation"""
    try:
        print("Testing DINO encoder...")
        
        from moge.model.modules import DINOv2Encoder
        
        encoder = DINOv2Encoder(
            backbone="dinov2_vitb14",
            intermediate_layers=[8, 16, 20, 24],
            dim_out=768
        )
        
        print("✓ DINO encoder created successfully")
        print(f"  - Features dim: {encoder.dim_features}")
        print(f"  - Number of features: {encoder.num_features}")
        
        return True
    except Exception as e:
        print(f"✗ DINO encoder failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dino_forward():
    """Test DINO forward pass"""
    try:
        print("Testing DINO forward pass...")
        
        from moge.model.modules import DINOv2Encoder
        
        encoder = DINOv2Encoder(
            backbone="dinov2_vitb14", 
            intermediate_layers=[8, 16, 20, 24],
            dim_out=768
        )
        
        # Create dummy input
        batch_size = 1
        height, width = 224, 224  # Small test
        rgb = torch.randn(batch_size, 3, height, width)
        
        token_rows = height // 14
        token_cols = width // 14
        
        print(f"Input shape: {rgb.shape}")
        print(f"Token grid: {token_rows}x{token_cols}")
        
        with torch.no_grad():
            features = encoder(rgb, token_rows, token_cols)
            
        print(f"✓ DINO forward pass successful")
        print(f"  - Output shape: {features.shape}")
        
        return True
    except Exception as e:
        print(f"✗ DINO forward failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing DINO-CubifyAnything integration...")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        exit(1)
    
    print()
    
    # Test DINO encoder
    if not test_dino_encoder():
        exit(1)
        
    print()
    
    # Test forward pass
    if not test_dino_forward():
        exit(1)
    
    print("=" * 50)
    print("✓ All basic tests passed!")
    print("\nNext: Test GPU functionality")
