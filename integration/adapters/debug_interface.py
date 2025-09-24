import sys
sys.path.append('/mnt/data/dino_cubify_integration/ml-cubifyanything')

import torch
from cubifyanything.cubify_transformer import make_cubify_transformer

def debug_original_interface():
    """Debug what original CubifyAnything backbone returns"""
    
    # Создай оригинальную модель
    model = make_cubify_transformer(
        dimension=768,
        depth_model=False,
        embed_dim=256
    )
    
    print("Original backbone type:", type(model.backbone.backbone))
    print("Backbone methods:", [m for m in dir(model.backbone.backbone) if not m.startswith('_')])
    
    # Создай тестовые данные в правильном формате
    batch_size = 1
    height, width = 512, 768
    
    # Правильный формат BatchedPosedSensor
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
    
    rgb_tensor = torch.randn(batch_size, 3, height, width)
    sensor = MockSensor(rgb_tensor)
    batched_sensor = {"image": sensor}
    
    # Тестируй оригинальный backbone
    with torch.no_grad():
        try:
            original_output = model.backbone.backbone(batched_sensor)
            print(f"Original output type: {type(original_output)}")
            print(f"Original output: {original_output}")
            
            if isinstance(original_output, dict):
                print("Output keys:", list(original_output.keys()))
                for key, value in original_output.items():
                    print(f"  {key}: {value.shape if hasattr(value, 'shape') else type(value)}")
            elif isinstance(original_output, list):
                print(f"Output list length: {len(original_output)}")
                for i, item in enumerate(original_output):
                    print(f"  [{i}]: {item.shape if hasattr(item, 'shape') else type(item)}")
                    
        except Exception as e:
            print(f"Original test failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    debug_original_interface()
