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
    """Adapter to replace CubifyAnything's ViT with MoGe's DINO-ViT v2"""
    
    def __init__(
        self,
        backbone: str = "dinov2_vitb14",
        embed_dim: int = 768,
        patch_size: int = 16,
        depth_modality: bool = True,
        image_name: str = "image", 
        depth_name: str = "depth"
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.depth_modality = depth_modality
        self.image_name = image_name
        self.depth_name = depth_name
        
        # DINO encoder
        self.dino_encoder = DINOv2Encoder(
            backbone=backbone,
            intermediate_layers=[8, 16, 20, 24],
            dim_out=embed_dim
        )
        
        # Patch size adapter: 14x14 -> 16x16
        self.patch_adapter = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GroupNorm(32, embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        )
        
        # CubifyAnything interface compatibility
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

    def forward(self, sensor: BatchedPosedSensor) -> Dict[str, torch.Tensor]:
        # Extract RGB image
        rgb = sensor[self.image_name].data.tensor
        batch_size, _, height, width = rgb.shape
        
        # Calculate token dimensions for DINO (patch size 14)
        token_rows = height // 14
        token_cols = width // 14
        
        # Get DINO features
        dino_features = self.dino_encoder(rgb, token_rows, token_cols)
        
        # Adapt features
        features = self.patch_adapter(dino_features)
        
        # Resize to target dimensions (stride 16)
        target_h = height // self.patch_size
        target_w = width // self.patch_size
        
        final_features = F.interpolate(
            features,
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False
        )
        
        return {"last_feat": final_features}


def test_adapter():
    """Test the adapter"""
    print("Testing DINO-CubifyAnything adapter...")
    
    # Create adapter
    adapter = DINOCubifyAdapter()
    print(f"Adapter created: {adapter.embed_dim}D, patch {adapter.patch_size}")
    
    # Test on CPU first
    batch_size = 1
    height, width = 224, 224
    
    # Create dummy sensor data
    class DummySensor:
        def __init__(self, tensor):
            self.tensor = tensor
        @property
        def data(self):
            return self
        @property
        def image_sizes(self):
            return [(height, width)] * batch_size
    
    rgb_tensor = torch.randn(batch_size, 3, height, width)
    sensor = {"image": DummySensor(rgb_tensor)}
    
    # Forward pass
    with torch.no_grad():
        outputs = adapter(sensor)
    
    print(f"CPU forward pass successful!")
    print(f"Output shape: {outputs['last_feat'].shape}")
    
    # Test GPU if available
    if torch.cuda.is_available():
        print("Testing GPU...")
        adapter_gpu = adapter.cuda()
        sensor_gpu = {"image": DummySensor(rgb_tensor.cuda())}
        
        with torch.no_grad():
            outputs_gpu = adapter_gpu(sensor_gpu)
        
        print(f"GPU forward pass successful!")
        print(f"GPU output shape: {outputs_gpu['last_feat'].shape}")
    
    return True


if __name__ == "__main__":
    test_adapter()
