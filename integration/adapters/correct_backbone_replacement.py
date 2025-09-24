import sys
sys.path.append('/mnt/data/dino_cubify_integration/ml-cubifyanything')
sys.path.append('/mnt/data/dino_cubify_integration/MoGe')

import torch
from dino_cubify_adapter_corrected import DINOCubifyAdapter
from cubifyanything.cubify_transformer import make_cubify_transformer

def correct_replacement():
    """Correct way to replace the backbone"""
    
    print("Creating model...")
    model = make_cubify_transformer(
        dimension=768,
        depth_model=False,
        embed_dim=256
    )
    
    print(f"Original: {type(model.backbone)}")
    print(f"Original backbone: {type(model.backbone.backbone)}")
    
    # Create DINO adapter
    dino_adapter = DINOCubifyAdapter()
    dino_adapter.dino_encoder.init_weights()
    
    # Proper replacement - replace the actual ViT inside the backbone
    original_backbone = model.backbone.backbone
    model.backbone.backbone = dino_adapter
    
    # Verify replacement worked
    print(f"After replacement: {type(model.backbone.backbone)}")
    print(f"Replacement successful: {type(model.backbone.backbone).__name__ == 'DINOCubifyAdapter'}")
    
    # Test the replaced model
    class MockImageData:
        def __init__(self, tensor, sizes):
            self.tensor = tensor
            self.image_sizes = sizes
    
    class MockImageSensor:
        def __init__(self, tensor, sizes):
            self.data = MockImageData(tensor, sizes)
            self.info = [None]
    
    rgb = torch.randn(1, 3, 224, 224)
    sensor = MockImageSensor(rgb, [(224, 224)])
    
    print(f"\nTesting replacement...")
    try:
        # Test the backbone directly first
        direct_result = model.backbone.backbone({"image": sensor})
        print(f"Direct backbone call: {type(direct_result)}")
        if isinstance(direct_result, dict):
            print(f"Keys: {list(direct_result.keys())}")
            print(f"Shape: {direct_result['last_feat'].shape}")
        
        # Now test through the full backbone (which includes Joiner wrapper)
        full_result = model.backbone({"image": sensor})
        print(f"Full backbone call: {type(full_result)}")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    correct_replacement()
