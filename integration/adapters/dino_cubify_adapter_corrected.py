import sys
sys.path.append('/mnt/data/dino_cubify_integration/ml-cubifyanything')
sys.path.append('/mnt/data/dino_cubify_integration/MoGe')

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from moge.model.modules import DINOv2Encoder

class DINOCubifyAdapter(nn.Module):
    """Corrected adapter with proper spatial dimensions"""
    
    def __init__(
        self,
        backbone: str = "dinov2_vitb14",
        embed_dim: int = 768,
        patch_size: int = 16,
        image_name: str = "image"
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.image_name = image_name
        
        # DINO encoder
        self.dino_encoder = DINOv2Encoder(
            backbone=backbone,
            intermediate_layers=[11],
            dim_out=embed_dim
        )
        
        # Patch size adapter with upsampling to match stride 16
        self.spatial_adapter = nn.Sequential(
            # DINO gives features at stride 14, we need stride 16
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GroupNorm(32, embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, embed_dim),
            nn.GELU()
        )
        
        # CubifyAnything interface
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
        # Extract image
        if hasattr(sensor, '__getitem__'):
            image_sensor = sensor[self.image_name]
            rgb = image_sensor.data.tensor
        else:
            rgb = sensor.data.tensor
            
        batch_size, _, height, width = rgb.shape
        
        # DINO processing with patch size 14
        token_rows = height // 14
        token_cols = width // 14
        
        # Get DINO features
        dino_features = self.dino_encoder(rgb, token_rows, token_cols)
        
        # Apply spatial adapter
        adapted_features = self.spatial_adapter(dino_features)
        
        # Calculate target dimensions for stride 16 (like original CubifyAnything)
        target_h = height // self.patch_size
        target_w = width // self.patch_size
        
        # Resize to exact target dimensions
        final_features = F.interpolate(
            adapted_features,
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False
        )
        
        return {"last_feat": final_features}


def test_size_matching():
    """Test that output sizes match original CubifyAnything"""
    
    adapter = DINOCubifyAdapter()
    
    # Test with the same size as debug_interface.py
    height, width = 512, 768
    
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
    
    rgb = torch.randn(1, 3, height, width)
    sensor = MockSensor(rgb)
    batched_sensor = {"image": sensor}
    
    with torch.no_grad():
        outputs = adapter(batched_sensor)
    
    expected_h = height // 16  # 32
    expected_w = width // 16   # 48
    
    print(f"Input size: {height}x{width}")
    print(f"Output shape: {outputs['last_feat'].shape}")
    print(f"Expected: [1, 768, {expected_h}, {expected_w}]")
    print(f"Matches original: {outputs['last_feat'].shape == torch.Size([1, 768, expected_h, expected_w])}")
    
    return outputs['last_feat'].shape == torch.Size([1, 768, expected_h, expected_w])

if __name__ == "__main__":
    success = test_size_matching()
    print(f"Size matching test: {'PASSED' if success else 'FAILED'}")
