import sys
sys.path.append('/mnt/data/dino_cubify_integration/ml-cubifyanything')
sys.path.append('/mnt/data/dino_cubify_integration/MoGe')

import torch
from dino_cubify_adapter_fixed import DINOCubifyAdapter
from cubifyanything.cubify_transformer import make_cubify_transformer

def create_dino_cubify_model():
    """Create full CubifyAnything model with DINO backbone"""
    print("Creating full DINO-CubifyAnything model...")
    
    # Создай базовую модель CubifyAnything
    model = make_cubify_transformer(
        dimension=768,  # DINO ViT-B dimension
        depth_model=False,  # Пока без depth
        embed_dim=256
    )
    
    print("Original backbone:", type(model.backbone.backbone))
    
    # Замени ViT backbone на DINO adapter
    dino_adapter = DINOCubifyAdapter(embed_dim=768)
    dino_adapter.dino_encoder.init_weights()  # Загрузи предобученные веса
    
    # Замени backbone
    model.backbone.backbone = dino_adapter
    
    print("Replaced with DINO adapter")
    print(f"Model device: {next(model.parameters()).device}")
    
    return model

def test_full_model():
    """Test complete model"""
    model = create_dino_cubify_model()
    model.eval()
    
    # Создай тестовые данные
    batch_size = 1
    height, width = 512, 768
    
    # Имитируй BatchedPosedSensor
    class MockImageSensor:
        def __init__(self, tensor, sizes):
            self.data = MockData(tensor)
            self.info = [None] * len(sizes)
        
        def __getitem__(self, key):
            return self.data if key == "tensor" else None
    
    class MockData:
        def __init__(self, tensor):
            self.tensor = tensor
            self.image_sizes = [(height, width)] * batch_size
    
    rgb_tensor = torch.randn(batch_size, 3, height, width)
    
    batched_sensor = {
        "wide": {
            "image": MockImageSensor(rgb_tensor, [(height, width)] * batch_size)
        }
    }
    
    print(f"Testing with input shape: {rgb_tensor.shape}")
    
    try:
        with torch.no_grad():
            # Тестируй только backbone
            features = model.backbone(batched_sensor["wide"])
            print(f"Backbone output: {features['last_feat'].shape}")
            print("Full integration test successful!")
            
    except Exception as e:
        print(f"Integration test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_full_model()
