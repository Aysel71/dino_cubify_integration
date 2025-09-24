import sys
sys.path.append('/mnt/data/dino_cubify_integration/MoGe')

import torch
from dino_cubify_adapter_fixed import DINOCubifyAdapter

def load_pretrained_weights():
    """Load pretrained DINO weights"""
    print("Loading DINO adapter with pretrained weights...")
    
    adapter = DINOCubifyAdapter()
    
    # Инициализируй DINO с предобученными весами
    print("Initializing DINO weights...")
    adapter.dino_encoder.init_weights()
    print("DINO pretrained weights loaded!")
    
    # Тестируй с весами
    rgb = torch.randn(1, 3, 224, 224)
    
    class DummySensor:
        def __init__(self, tensor):
            self.tensor = tensor
        @property
        def data(self):
            return self
    
    sensor = {"image": DummySensor(rgb)}
    
    with torch.no_grad():
        outputs = adapter(sensor)
    
    print(f"Forward pass with pretrained weights successful!")
    print(f"Output shape: {outputs['last_feat'].shape}")
    
    return adapter

if __name__ == "__main__":
    adapter = load_pretrained_weights()
