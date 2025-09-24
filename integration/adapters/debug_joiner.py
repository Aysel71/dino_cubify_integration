import sys
sys.path.append('/mnt/data/dino_cubify_integration/ml-cubifyanything')

import torch
from cubifyanything.cubify_transformer import make_cubify_transformer, Joiner
from dino_cubify_adapter_corrected import DINOCubifyAdapter

def debug_joiner_class():
    """Debug how Joiner works"""
    
    model = make_cubify_transformer(
        dimension=768,
        depth_model=False,
        embed_dim=256
    )
    
    joiner = model.backbone
    print(f"Joiner type: {type(joiner)}")
    print(f"Joiner is Sequential: {isinstance(joiner, torch.nn.Sequential)}")
    
    # Check if Joiner inherits from Sequential
    print(f"Joiner MRO: {type(joiner).__mro__}")
    
    # See what's inside Joiner
    print(f"Joiner length: {len(joiner)}")
    for i, module in enumerate(joiner):
        print(f"  [{i}]: {type(module)}")
    
    # Check if backbone is a property
    print(f"backbone is property: {isinstance(type(joiner).backbone, property) if hasattr(type(joiner), 'backbone') else 'No backbone property'}")
    
    # Try direct indexing
    print(f"joiner[0]: {type(joiner[0])}")
    
    # Try replacement via indexing
    print(f"\nTrying replacement via indexing...")
    dino_adapter = DINOCubifyAdapter()
    
    original_id = id(joiner[0])
    joiner[0] = dino_adapter
    new_id = id(joiner[0])
    
    print(f"Original ViT id: {original_id}")
    print(f"New module id: {new_id}")
    print(f"New module type: {type(joiner[0])}")
    print(f"Replacement successful: {type(joiner[0]).__name__ == 'DINOCubifyAdapter'}")
    
    # Also try accessing via .backbone
    print(f"Via .backbone: {type(joiner.backbone)}")
    
    return joiner

if __name__ == "__main__":
    joiner = debug_joiner_class()
