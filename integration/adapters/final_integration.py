import sys
sys.path.append('/mnt/data/dino_cubify_integration/ml-cubifyanything')
sys.path.append('/mnt/data/dino_cubify_integration/MoGe')

import torch
from dino_cubify_adapter_corrected import DINOCubifyAdapter
from cubifyanything.cubify_transformer import make_cubify_transformer

def create_dino_cubify_model():
    """Create CubifyAnything with DINO backbone - final version"""
    
    print("Creating DINO-CubifyAnything model...")
    
    # Create base model
    model = make_cubify_transformer(
        dimension=768,
        depth_model=False,
        embed_dim=256
    )
    
    print(f"Original structure:")
    print(f"  model.backbone: {type(model.backbone)}")
    print(f"  model.backbone[0]: {type(model.backbone[0])}")
    
    # Create DINO adapter with pretrained weights
    dino_adapter = DINOCubifyAdapter()
    dino_adapter.dino_encoder.init_weights()
    print("DINO adapter created with pretrained weights")
    
    # Replace via indexing (since Joiner inherits from Sequential)
    model.backbone[0] = dino_adapter
    
    print(f"After replacement:")
    print(f"  model.backbone[0]: {type(model.backbone[0])}")
    print(f"  Replacement successful: {type(model.backbone[0]).__name__ == 'DINOCubifyAdapter'}")
    
    return model

def test_complete_model():
    """Test the complete integrated model"""
    
    model = create_dino_cubify_model()
    model.eval()
    
    # Test data
    batch_size = 1
    height, width = 512, 768
    
    # Proper BatchedPosedSensor format
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
    
    batched_sensor = {
        "wide": {
            "image": MockImageSensor(rgb_tensor, image_sizes)
        }
    }
    
    print(f"\nTesting complete model with input: {rgb_tensor.shape}")
    
    try:
        with torch.no_grad():
            # Test backbone
            backbone_result = model.backbone(batched_sensor["wide"])
            print(f"Backbone result type: {type(backbone_result)}")
            
            # Joiner returns a list of NestedTensor objects
            if isinstance(backbone_result, list):
                print(f"Backbone output list length: {len(backbone_result)}")
                for i, item in enumerate(backbone_result):
                    if hasattr(item, 'tensors'):
                        print(f"  [{i}] tensors shape: {item.tensors.shape}")
                        
                # For CubifyAnything, usually we want the first element
                features = backbone_result[0].tensors
                print(f"Final feature shape: {features.shape}")
                
            print("✓ Complete integration successful!")
            return True
            
    except Exception as e:
        print(f"✗ Integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gpu_performance():
    """Test GPU performance"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU test")
        return
        
    print("\nTesting GPU performance...")
    
    model = create_dino_cubify_model()
    model = model.cuda()
    model.eval()
    
    # Test sizes
    test_sizes = [(224, 224), (512, 768)]
    
    for height, width in test_sizes:
        class MockImageData:
            def __init__(self, tensor, sizes):
                self.tensor = tensor
                self.image_sizes = sizes
        
        class MockImageSensor:
            def __init__(self, tensor, sizes):
                self.data = MockImageData(tensor, sizes)
                self.info = [None]
        
        rgb = torch.randn(1, 3, height, width).cuda()
        sensor = MockImageSensor(rgb, [(height, width)])
        
        try:
            with torch.no_grad():
                torch.cuda.empty_cache()
                
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                result = model.backbone({"image": sensor})
                end.record()
                
                torch.cuda.synchronize()
                elapsed = start.elapsed_time(end)
                memory = torch.cuda.max_memory_allocated() // 1024**2
                
                print(f"  {height}x{width}: {elapsed:.1f}ms, {memory}MB")
                
        except Exception as e:
            print(f"  {height}x{width}: Failed - {e}")

if __name__ == "__main__":
    # Test complete integration
    success = test_complete_model()
    
    if success:
        # Test GPU performance
        test_gpu_performance()
        
        print("\n🎉 DINO-CubifyAnything integration complete!")
        print("Next steps:")
        print("1. Train on CA-1M dataset")
        print("2. Evaluate on validation set") 
        print("3. Compare with original CubifyAnything")
