import sys
sys.path.append('/mnt/data/dino_cubify_integration/ml-cubifyanything')

import torch
from cubifyanything.cubify_transformer import make_cubify_transformer

def debug_model_structure():
    """Debug the full model structure"""
    
    model = make_cubify_transformer(
        dimension=768,
        depth_model=False,
        embed_dim=256
    )
    
    print("Model structure:")
    print(f"model type: {type(model)}")
    print(f"model.backbone type: {type(model.backbone)}")
    print(f"model.backbone.backbone type: {type(model.backbone.backbone)}")
    
    # Check if there are multiple levels
    print(f"\nmodel.backbone attributes:")
    for attr in dir(model.backbone):
        if not attr.startswith('_'):
            obj = getattr(model.backbone, attr)
            if hasattr(obj, '__class__'):
                print(f"  {attr}: {type(obj)}")
    
    print(f"\nmodel.backbone methods:")
    backbone_methods = [m for m in dir(model.backbone) if callable(getattr(model.backbone, m)) and not m.startswith('_')]
    print(backbone_methods)
    
    # Test what the backbone actually returns
    class MockData:
        def __init__(self, tensor):
            self.tensor = tensor
    
    class MockSensor:
        def __init__(self, tensor):
            self.data = MockData(tensor)
    
    rgb = torch.randn(1, 3, 224, 224)
    sensor = MockSensor(rgb)
    batched_sensor = {"image": sensor}
    
    print(f"\nTesting model.backbone() call:")
    try:
        result = model.backbone(batched_sensor)
        print(f"model.backbone() returns: {type(result)}")
        if isinstance(result, list):
            print(f"List length: {len(result)}")
            for i, item in enumerate(result):
                print(f"  [{i}]: {type(item)}")
                if hasattr(item, 'tensors'):
                    print(f"    tensors shape: {item.tensors.shape}")
        elif isinstance(result, dict):
            print(f"Dict keys: {list(result.keys())}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_model_structure()
