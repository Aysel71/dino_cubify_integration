import sys
sys.path.append('/mnt/data/dino_cubify_integration/ml-cubifyanything')
sys.path.append('/mnt/data/dino_cubify_integration/MoGe')

import torch
from dino_cubify_adapter_fixed import DINOCubifyAdapter

def test_realistic_sizes():
    """Test with CubifyAnything realistic image sizes"""
    
    adapter = DINOCubifyAdapter().cuda()
    
    # Типичные размеры для CubifyAnything
    test_sizes = [
        (512, 768),   # small
        (768, 1024),  # medium  
        (1024, 1280), # large
    ]
    
    for height, width in test_sizes:
        print(f"\nTesting {height}x{width}...")
        
        class DummySensor:
            def __init__(self, tensor):
                self.tensor = tensor
            @property
            def data(self):
                return self
        
        rgb = torch.randn(1, 3, height, width).cuda()
        sensor = {"image": DummySensor(rgb)}
        
        try:
            with torch.no_grad():
                torch.cuda.empty_cache()
                
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                outputs = adapter(sensor)
                end.record()
                
                torch.cuda.synchronize()
                elapsed = start.elapsed_time(end)
                memory = torch.cuda.max_memory_allocated() // 1024**2
                
                print(f"  Output: {outputs['last_feat'].shape}")
                print(f"  Time: {elapsed:.1f}ms")
                print(f"  Memory: {memory}MB")
                
        except Exception as e:
            print(f"  Failed: {e}")

if __name__ == "__main__":
    test_realistic_sizes()
