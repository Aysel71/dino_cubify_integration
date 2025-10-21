#!/usr/bin/env python3
"""
Create SUN RGB-D dataset with REAL 3D annotations from .mat files
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import scipy.io as sio
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_sunrgbd_with_real_annotations(data_root: str, max_scenes: int = 500):
    """Load SUN RGB-D with real 3D annotations from .mat file"""
    
    data_path = Path(data_root)
    processed_dir = data_path / "processed"
    processed_dir.mkdir(exist_ok=True)
    
    # Load metadata
    meta_file = data_path / "SUNRGBDMeta3DBB_v2.mat"
    if not meta_file.exists():
        logger.error(f"Metadata file not found: {meta_file}")
        return
    
    logger.info(f"Loading metadata from {meta_file}")
    metadata = sio.loadmat(str(meta_file))
    sun_rgbd_meta = metadata['SUNRGBDMeta'][0]
    
    logger.info(f"Found {len(sun_rgbd_meta)} scenes in metadata")
    
    # Class mapping (SUN RGB-D classes to indices)
    class_names = [
        'bathtub', 'bed', 'bookshelf', 'box', 'chair', 'counter', 'desk',
        'door', 'dresser', 'garbage_bin', 'lamp', 'monitor', 'night_stand',
        'pillow', 'sink', 'sofa', 'table', 'tv', 'toilet'
    ]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    valid_scenes = []
    
    logger.info(f"Processing up to {max_scenes} scenes...")
    
    for i, scene_meta in enumerate(tqdm(sun_rgbd_meta[:max_scenes * 2], desc="Processing scenes")):
        if len(valid_scenes) >= max_scenes:
            break
            
        try:
            # Parse scene metadata - Field 0 contains sequence name
            sequence_name = str(scene_meta[0][0])  # Extract string from numpy array
            scene_name = Path(sequence_name).name
            
            # Construct file paths - add data_path to sequence_name  
            rgb_path = data_path / sequence_name / "image"
            depth_path = data_path / sequence_name / "depth"
            
            # Find actual RGB file
            if not rgb_path.exists():
                continue
            
            jpg_files = list(rgb_path.glob("*.jpg"))
            if not jpg_files:
                continue
            
            rgb_file = jpg_files[0]
            
            # Find depth file 
            depth_file = None
            if depth_path.exists():
                png_files = list(depth_path.glob("*.png"))
                if png_files:
                    depth_file = png_files[0]
            
            # Parse 3D boxes and labels from Field 10
            boxes_3d = []
            labels = []
            
            if len(scene_meta) > 10 and scene_meta[10].size > 0:  # Field 10 contains 3D objects
                objects = scene_meta[10][0]  # Array of objects
                
                for obj in objects:
                    try:
                        # Each object is a tuple with: (rotation, dimensions, center, class_name, ...)
                        if len(obj) >= 4:
                            dimensions = obj[1][0]  # [w, h, d]  
                            center = obj[2][0]      # [x, y, z]
                            class_name = str(obj[3][0])  # class name
                            
                            # Clean class name
                            class_name = class_name.lower().strip()
                            
                            # Map to our classes
                            mapped_class = None
                            for valid_class in class_names:
                                if valid_class in class_name or class_name in valid_class:
                                    mapped_class = valid_class
                                    break
                            
                            # Try common mappings
                            if not mapped_class:
                                if any(word in class_name for word in ['chair', 'stool']):
                                    mapped_class = 'chair'
                                elif any(word in class_name for word in ['table', 'desk']):
                                    mapped_class = 'table'
                                elif 'bed' in class_name:
                                    mapped_class = 'bed'
                                elif 'shelf' in class_name:
                                    mapped_class = 'bookshelf'
                                elif 'sofa' in class_name:
                                    mapped_class = 'sofa'
                                elif 'lamp' in class_name:
                                    mapped_class = 'lamp'
                                elif 'dresser' in class_name:
                                    mapped_class = 'dresser'
                                elif 'night_stand' in class_name:
                                    mapped_class = 'night_stand'
                                else:
                                    mapped_class = 'box'  # Default
                            
                            if mapped_class and mapped_class in class_to_idx:
                                # Create 3D box in [cx, cy, cz, w, h, l, ry] format
                                if len(center) >= 3 and len(dimensions) >= 3:
                                    cx, cy, cz = center[:3]
                                    w, h, l = dimensions[:3] 
                                    ry = 0.0  # Simplified rotation
                                    
                                    box_3d = [float(cx), float(cy), float(cz), 
                                             float(w), float(h), float(l), float(ry)]
                                    label = class_to_idx[mapped_class]
                                    
                                    boxes_3d.append(box_3d)
                                    labels.append(label)
                                    
                    except Exception as e:
                        logger.debug(f"Error processing object: {e}")
                        continue
            
            # If no valid boxes found, skip this scene (don't use dummy)
            if len(boxes_3d) == 0:
                continue
            
            # Get camera intrinsics from Field 2
            intrinsics = None
            if len(scene_meta) > 2 and scene_meta[2].size > 0:
                try:
                    intrinsics = scene_meta[2].tolist()
                except:
                    intrinsics = [[529.5, 0., 365.], [0., 529.5, 265.], [0., 0., 1.]]
            else:
                intrinsics = [[529.5, 0., 365.], [0., 529.5, 265.], [0., 0., 1.]]
            
            # Create scene data
            scene_data = {
                'scene_idx': len(valid_scenes),
                'scene_name': scene_name,
                'sequence_name': sequence_name,
                'rgb_path': str(rgb_file.relative_to(data_path)),
                'depth_path': str(depth_file.relative_to(data_path)) if depth_file else None,
                'boxes_3d': boxes_3d,
                'labels': labels,
                'num_boxes': len(boxes_3d),
                'intrinsics': intrinsics,
                'sensor_type': sequence_name.split('/')[1] if '/' in sequence_name else 'unknown'
            }
            
            valid_scenes.append(scene_data)
            
        except Exception as e:
            logger.debug(f"Error processing scene {i}: {e}")
            continue
    
    logger.info(f"Created {len(valid_scenes)} valid scenes with real annotations")
    
    # Create train/test split
    import random
    random.seed(42)
    valid_scenes_copy = valid_scenes.copy()
    random.shuffle(valid_scenes_copy)
    
    train_size = int(len(valid_scenes_copy) * 0.8)
    train_scenes = valid_scenes_copy[:train_size]
    test_scenes = valid_scenes_copy[train_size:]
    
    logger.info(f"Split: {len(train_scenes)} train, {len(test_scenes)} test")
    
    # Save data
    train_file = processed_dir / "train_scenes_with_annotations.json"
    test_file = processed_dir / "test_scenes_with_annotations.json"
    
    with open(train_file, 'w') as f:
        json.dump(train_scenes, f, indent=2)
    
    with open(test_file, 'w') as f:
        json.dump(test_scenes, f, indent=2)
    
    # Replace old files
    old_train = processed_dir / "train_scenes.json"
    old_test = processed_dir / "test_scenes.json"
    
    if old_train.exists():
        old_train.rename(processed_dir / "train_scenes_backup.json")
    if old_test.exists():
        old_test.rename(processed_dir / "test_scenes_backup.json")
    
    import shutil
    shutil.copy2(train_file, old_train)
    shutil.copy2(test_file, old_test)
    
    # Display samples
    logger.info("Sample scenes with real annotations:")
    for i, scene in enumerate(train_scenes[:3]):
        logger.info(f"Sample {i+1}: {scene['scene_name']}")
        logger.info(f"  Sensor: {scene['sensor_type']}")
        logger.info(f"  Boxes: {scene['num_boxes']}")
        logger.info(f"  Labels: {scene['labels']}")
        if scene['boxes_3d']:
            logger.info(f"  First box: {scene['boxes_3d'][0]}")
            class_name = class_names[scene['labels'][0]]
            logger.info(f"  First class: {class_name}")
    
    # Statistics
    stats = {
        'total_scenes': len(valid_scenes),
        'train_scenes': len(train_scenes),
        'test_scenes': len(test_scenes),
        'classes': {},
        'sensors': {}
    }
    
    # Count classes and sensors
    for scene in valid_scenes:
        for label in scene['labels']:
            class_name = class_names[label]
            stats['classes'][class_name] = stats['classes'].get(class_name, 0) + 1
        
        sensor = scene['sensor_type']
        stats['sensors'][sensor] = stats['sensors'].get(sensor, 0) + 1
    
    stats_file = processed_dir / "dataset_stats_with_annotations.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Statistics saved to {stats_file}")
    logger.info("Dataset ready for training with real 3D annotations!")
    
    return valid_scenes

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, 
                       default='/mnt/data/datasets/sun_rgbd')
    parser.add_argument('--max_scenes', type=int, default=500)
    
    args = parser.parse_args()
    
    load_sunrgbd_with_real_annotations(args.data_root, args.max_scenes)
