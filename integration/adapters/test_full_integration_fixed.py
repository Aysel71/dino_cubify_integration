import sys
sys.path.append('/mnt/data/dino_cubify_integration/ml-cubifyanything')
sys.path.append('/mnt/data/dino_cubify_integration/MoGe')

import torch
from dino_cubify_adapter_corrected import DINOCubifyAdapter
from cubifyanything.cubify_transformer import make_cubify_transformer

def test_full_replacement():
    """Test complete backbone replacement"""
    
    print("Creating CubifyAnything model with DINO backbone...")
    
    # Create original model
    model = make_cubify_transformer(
        dimension=768,
        depth_model=False,
        embed_dim=256
    )
    
    print(f"Original backbone: {type(model.backbone.backbone)}")
    
    # Replace with DINO adapter
    dino_adapter = DINOCubifyAdapter()
    dino_adapter.dino_encoder.init_weights()  # Load pretrained weights
    
    model.backbone.backbone = dino_adapter
    print(f"Replaced with: {type(model.backbone.backbone)}")
    
    # Test data
    batch_size = 1
    height, width = 512, 768
    
    class MockImageData:
        def __init__(self, tensor, sizes):
            self.tensor = tensor
            self.image_sizes = sizes
    
    class MockImageSensor:
        def __init__(self, tensor, sizes):
            self.data = MockImageData(tensor, sizes)
            self.info = [None] * len(sizes)
    
    rgb_tensor = torch.randn(batch_size, 3, height, width)
    image_sizes = [(height, width)] * batch_size
    
    # Create batched sensor in correct format
    batched_sensor = {
        "wide": {
            "image": MockImageSensor(rgb_tensor, image_sizes)
        }
    }
    
    print(f"Testing with input: {rgb_tensor.shape}")
    
    try:
        with torch.no_grad():
            # Test backbone only
            backbone_output = model.backbone(batched_sensor["wide"])
            print(f"Backbone output shape: {backbone_output['last_feat'].shape}")
            print("✓ Backbone replacement successful!")
            
            return True
            
    except Exception as e:
        print(f"✗ Integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_full_replacement()
