import sys
sys.path.append('/mnt/data/dino_cubify_integration/ml-cubifyanything')
sys.path.append('/mnt/data/dino_cubify_integration/MoGe')

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union

from moge.model.modules import DINOv2Encoder
from cubifyanything.batching import BatchedPosedSensor

class DINOCubifyAdapter(nn.Module):
    """Final adapter that matches CubifyAnything interface exactly"""
    
    def __init__(
        self,
        backbone: str = "dinov2_vitb14",
        embed_dim: int = 768,
        patch_size: int = 16,
        depth_modality: bool = False,
        image_name: str = "image"
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.depth_modality = depth_modality
        self.image_name = image_name
        
        # DINO encoder
        self.dino_encoder = DINOv2Encoder(
            backbone=backbone,
            intermediate_layers=[11],  # Last layer only
            dim_out=embed_dim
        )
        
        # Patch adapter
        self.patch_adapter = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GroupNorm(32, embed_dim),
            nn.GELU()
        )
        
        # Copy all attributes from original ViT for compatibility
        self._out_feature_channels = {"last_feat": embed_dim}
        self._out_feature_strides = {"last_feat": patch_size}
        self._out_features = ["last_feat"]
        self._square_pad = [256, 384, 512, 640, 768, 896, 1024, 1280]

    @property
    def num_channels(self):
        return list(self._out_feature_channels.values())

    @property
    def size_divisibility(self):
        return self.patch_size

    def forward(self, sensor) -> Dict[str, torch.Tensor]:
        """Forward pass matching original ViT interface"""
        
        # Extract image tensor - handle different input formats
        if hasattr(sensor, '__getitem__'):
            image_sensor = sensor[self.image_name]
            rgb = image_sensor.data.tensor
        else:
            rgb = sensor.data.tensor
            
        batch_size, _, height, width = rgb.shape
        
        # DINO processing
        token_rows = height // 14
        token_cols = width // 14
        
        # Get DINO features
        dino_features = self.dino_encoder(rgb, token_rows, token_cols)
        
        # Process features
        features = self.patch_adapter(dino_features)
        
        # Resize to match expected output stride
        target_h = height // self.patch_size
        target_w = width // self.patch_size
        
        final_features = F.interpolate(
            features,
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False
        )
        
        # Return in expected format
        return {"last_feat": final_features}


def test_compatibility():
    """Test compatibility with CubifyAnything"""
    
    # Test standalone adapter first
    adapter = DINOCubifyAdapter()
    
    class MockData:
        def __init__(self, tensor):
            self.tensor = tensor
    
    class MockSensor:
        def __init__(self, tensor):
            self.data = MockData(tensor)
        
        def __getitem__(self, key):
            if key == "image":
                return self
            return None
    
    rgb = torch.randn(1, 3, 224, 224)
    sensor = MockSensor(rgb)
    batched_sensor = {"image": sensor}
    
    with torch.no_grad():
        outputs = adapter(batched_sensor)
    
    print(f"Adapter test successful!")
    print(f"Output: {outputs}")
    print(f"Shape: {outputs['last_feat'].shape}")

if __name__ == "__main__":
    test_compatibility()
