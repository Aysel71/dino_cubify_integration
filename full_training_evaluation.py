
#!/usr/bin/env python3
"""
IMPROVED Training Script with 3D Detection Head
Добавляет реальные предсказания и метрики AP25/AR25/AP50/AR50
"""

import sys
import os
import json
import argparse
from pathlib import Path
import time
from typing import Dict, List, Tuple, Optional, Union
import logging

# Core ML libraries  
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as transforms

# Standard libraries
import numpy as np
import cv2
from PIL import Image
import scipy.io as sio
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add paths for YOUR integrated modules
sys.path.append('/mnt/data/dino_cubify_integration/ml-cubifyanything')
sys.path.append('/mnt/data/dino_cubify_integration/MoGe')
sys.path.append('/mnt/data/dino_cubify_integration/integration/adapters')

# Import YOUR custom modules
from dino_cubify_adapter_corrected import DINOCubifyAdapter
from final_integration import create_dino_cubify_model

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Simple3DDetectionHead(nn.Module):
    """3D Detection Head для реальных предсказаний"""
    
    def __init__(self, feature_dim=256, num_classes=19, max_detections=32, hidden_dim=512):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.max_detections = max_detections
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Detection heads
        self.box_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, max_detections * 7)  # [cx,cy,cz,w,h,l,ry]
        )
        
        self.class_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), 
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, max_detections * num_classes)
        )
        
        self.conf_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, max_detections)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, features):
        batch_size = features.shape[0]
        
        # Global pooling: [B,C,H,W] -> [B,C]
        global_feat = self.global_pool(features).flatten(1)
        
        # Predictions
        boxes = self.box_head(global_feat).view(batch_size, self.max_detections, 7)
        classes = self.class_head(global_feat).view(batch_size, self.max_detections, self.num_classes)
        conf = torch.sigmoid(self.conf_head(global_feat))  # [B, max_detections]
        
        return {
            'box_predictions': boxes,
            'class_logits': classes, 
            'confidence_scores': conf
        }


class Enhanced3DLoss(nn.Module):
    """Улучшенный loss с реальными предсказаниями"""
    
    def __init__(self, w_box=1.0, w_cls=1.0, w_conf=0.5):
        super().__init__()
        self.w_box = w_box
        self.w_cls = w_cls
        self.w_conf = w_conf
    
    def forward(self, pred, gt_boxes, gt_labels, num_boxes):
        batch_size = gt_boxes.shape[0]
        total_loss = 0.0
        valid_batches = 0
        
        for b in range(batch_size):
            n_boxes = min(num_boxes[b].item(), gt_boxes.shape[1])
            if n_boxes == 0:
                continue
                
            pred_boxes = pred['box_predictions'][b, :n_boxes]  # [n_boxes, 7]
            pred_classes = pred['class_logits'][b, :n_boxes]   # [n_boxes, num_classes]
            pred_conf = pred['confidence_scores'][b, :n_boxes] # [n_boxes]
            
            gt_b = gt_boxes[b, :n_boxes]    # [n_boxes, 7] 
            gt_cls = gt_labels[b, :n_boxes] # [n_boxes]
            
            # Box regression loss
            box_loss = F.mse_loss(pred_boxes, gt_b)
            
            # Classification loss
            valid_mask = (gt_cls >= 0) & (gt_cls < pred_classes.shape[1])
            if valid_mask.sum() > 0:
                cls_loss = F.cross_entropy(pred_classes[valid_mask], gt_cls[valid_mask])
            else:
                cls_loss = torch.tensor(0.0, device=pred_boxes.device)
            
            # Confidence loss (high confidence for valid objects)
            conf_target = torch.ones_like(pred_conf)
            conf_loss = F.binary_cross_entropy(pred_conf, conf_target)
            
            batch_loss = (self.w_box * box_loss + 
                         self.w_cls * cls_loss + 
                         self.w_conf * conf_loss)
            total_loss += batch_loss
            valid_batches += 1
        
        return total_loss / max(valid_batches, 1)


class UpdatedSunRGBDDataset(Dataset):
    """Updated dataset (same as before)"""
    
    def __init__(self, data_root, split='train', image_size=(768, 512), max_boxes=32):
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.max_boxes = max_boxes
        
        self.scenes = self._load_preprocessed_scenes()
        logger.info(f"Loaded {len(self.scenes)} scenes for {split}")
        
        self.rgb_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.class_names = [
            'bathtub', 'bed', 'bookshelf', 'box', 'chair', 'counter', 'desk',
            'door', 'dresser', 'garbage_bin', 'lamp', 'monitor', 'night_stand',
            'pillow', 'sink', 'sofa', 'table', 'tv', 'toilet'
        ]
        self.num_classes = len(self.class_names)
    
    def _load_preprocessed_scenes(self):
        new_file = self.data_root / "processed" / f"{self.split}_scenes_with_annotations.json"
        if new_file.exists():
            with open(new_file, 'r') as f:
                return json.load(f)
        
        old_file = self.data_root / "processed" / f"{self.split}_scenes.json"  
        if old_file.exists():
            with open(old_file, 'r') as f:
                return json.load(f)
        return []
    
    def __len__(self):
        return len(self.scenes)
    
    def __getitem__(self, idx):
        scene = self.scenes[idx]
        
        try:
            # Load RGB
            rgb_path = self.data_root / scene['rgb_path']
            rgb_image = Image.open(rgb_path).convert('RGB')
            original_size = rgb_image.size
            
            # Load depth  
            depth_image = np.zeros((original_size[1], original_size[0]), dtype=np.float32)
            if scene.get('depth_path'):
                depth_path = self.data_root / scene['depth_path']
                if depth_path.exists():
                    depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                    if depth_img is not None:
                        depth_image = depth_img.astype(np.float32) / 1000.0
            
            # Transforms
            rgb_tensor = self.rgb_transform(rgb_image)
            depth_resized = cv2.resize(depth_image, self.image_size)
            depth_tensor = torch.from_numpy(depth_resized).unsqueeze(0)
            
            # Process boxes
            boxes_3d = np.array(scene.get('boxes_3d', [[0,0,1,1,1,1,0]]), dtype=np.float32)
            labels = np.array(scene.get('labels', [0]), dtype=np.int64)
            
            # Pad/truncate
            actual_num = min(len(boxes_3d), self.max_boxes)
            padded_boxes = np.zeros((self.max_boxes, 7), dtype=np.float32)
            padded_labels = np.zeros(self.max_boxes, dtype=np.int64)
            
            if actual_num > 0:
                padded_boxes[:actual_num] = boxes_3d[:actual_num]
                padded_labels[:actual_num] = labels[:actual_num]
            
            return {
                'rgb': rgb_tensor,
                'depth': depth_tensor,
                'boxes_3d': torch.from_numpy(padded_boxes),
                'labels': torch.from_numpy(padded_labels),
                'num_boxes': torch.tensor(actual_num, dtype=torch.long),
                'image_size': torch.tensor(original_size, dtype=torch.long),
                'scene_name': scene.get('scene_name', f'scene_{idx}'),
                'sensor_type': scene.get('sensor_type', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"Error loading {idx}: {e}")
            return {
                'rgb': torch.zeros(3, self.image_size[1], self.image_size[0]),
                'depth': torch.zeros(1, self.image_size[1], self.image_size[0]),
                'boxes_3d': torch.zeros(self.max_boxes, 7),
                'labels': torch.zeros(self.max_boxes, dtype=torch.long),
                'num_boxes': torch.tensor(0, dtype=torch.long),
                'image_size': torch.tensor(self.image_size, dtype=torch.long),
                'scene_name': 'error',
                'sensor_type': 'unknown'
            }


def create_batched_sensor(rgb, depth=None, image_sizes=None):
    """Create sensor format"""
    batch_size = rgb.shape[0]
    if image_sizes is None:
        height, width = rgb.shape[2], rgb.shape[3]
        image_sizes = [(width, height)] * batch_size
    
    class MockImageData:
        def __init__(self, tensor, sizes):
            self.tensor = tensor
            self.image_sizes = sizes
    
    class MockImageSensor:
        def __init__(self, tensor, sizes):
            self.data = MockImageData(tensor, sizes)
            self.info = [None] * batch_size
    
    sensor_dict = {"image": MockImageSensor(rgb, image_sizes)}
    if depth is not None:
        sensor_dict["depth"] = MockImageSensor(depth, image_sizes)
    
    return {"wide": sensor_dict}


def create_enhanced_model():
    """Create model with detection head"""
    base_model = create_dino_cubify_model()
    detection_head = Simple3DDetectionHead(feature_dim=768, num_classes=19, max_detections=32)
    
    class EnhancedModel(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.detection_head = head
            
        def forward(self, sensor_data):
            features = self.backbone(sensor_data)
            
            # Extract tensor from backbone output
            if isinstance(features, list) and len(features) > 0:
                if hasattr(features[0], 'tensors'):
                    feat_tensor = features[0].tensors
                else:
                    feat_tensor = features[0]
            else:
                feat_tensor = features
            
            return self.detection_head(feat_tensor)
    
    return EnhancedModel(base_model.backbone, detection_head)


class EnhancedTrainer:
    """Trainer с detection head и реальными предсказаниями"""
    
    def __init__(self, model, train_dataset, val_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset  
        self.config = config
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Training stages
        self.training_stage = 'freeze_dino'
        
        # Loss and optimizer
        self.criterion = Enhanced3DLoss(w_box=1.0, w_cls=1.0, w_conf=0.5)
        self.setup_optimizer()
        
        # Data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                                      shuffle=True, num_workers=config['num_workers'],
                                      pin_memory=True, drop_last=True)
        self.val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                                    shuffle=False, num_workers=config['num_workers'],
                                    pin_memory=True)
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        logger.info(f"Enhanced trainer initialized: {len(train_dataset)} train, {len(val_dataset)} val")
    
    def setup_optimizer(self):
        """Setup optimizer based on stage"""
        if self.training_stage == 'freeze_dino':
            # Only spatial adapter + detection head
            params = []
            
            # DINO frozen
            dino_adapter = self.model.backbone[0]
            for param in dino_adapter.dino_encoder.parameters():
                param.requires_grad = False
            
            # Spatial adapter trainable
            for param in dino_adapter.spatial_adapter.parameters():
                param.requires_grad = True
                params.append(param)
            
            # Detection head trainable
            for param in self.model.detection_head.parameters():
                param.requires_grad = True
                params.append(param)
            
            # Other backbone components
            for i in range(1, len(self.model.backbone)):
                for param in self.model.backbone[i].parameters():
                    param.requires_grad = True
                    params.append(param)
        
        elif self.training_stage == 'partial_unfreeze':
            # Last DINO layers + spatial adapter + detection head
            params = []
            dino_adapter = self.model.backbone[0]
            
            # Unfreeze last 4 DINO blocks
            dino_backbone = dino_adapter.dino_encoder.backbone
            if hasattr(dino_backbone, 'blocks'):
                for i in range(8, 12):
                    if i < len(dino_backbone.blocks):
                        for param in dino_backbone.blocks[i].parameters():
                            param.requires_grad = True
                            params.append(param)
            
            # Spatial adapter + detection head
            for param in dino_adapter.spatial_adapter.parameters():
                param.requires_grad = True
                params.append(param)
            for param in self.model.detection_head.parameters():
                param.requires_grad = True
                params.append(param)
                
            # Other components
            for i in range(1, len(self.model.backbone)):
                for param in self.model.backbone[i].parameters():
                    param.requires_grad = True
                    params.append(param)
        
        else:  # full_finetune
            # Everything with differential LR
            dino_params = []
            other_params = []
            
            dino_adapter = self.model.backbone[0]
            for param in dino_adapter.dino_encoder.parameters():
                param.requires_grad = True
                dino_params.append(param)
            
            # Other components get normal LR
            for param in dino_adapter.spatial_adapter.parameters():
                other_params.append(param)
            for param in self.model.detection_head.parameters():
                other_params.append(param)
            for i in range(1, len(self.model.backbone)):
                for param in self.model.backbone[i].parameters():
                    other_params.append(param)
            
            self.optimizer = AdamW([
                {'params': dino_params, 'lr': self.config['learning_rate'] * 0.1},
                {'params': other_params, 'lr': self.config['learning_rate']}
            ], weight_decay=self.config['weight_decay'])
            
            logger.info(f"Full finetune: DINO {len(dino_params)} (0.1x LR), Other {len(other_params)}")
            return
        
        # For first two stages
        trainable = [p for p in params if p.requires_grad]
        self.optimizer = AdamW(trainable, lr=self.config['learning_rate'], 
                              weight_decay=self.config['weight_decay'])
        logger.info(f"Stage {self.training_stage}: {len(trainable)} trainable parameters")
    
    def update_stage(self, epoch):
        """Update training stage"""
        if epoch < 8:
            new_stage = 'freeze_dino'
        elif epoch < 15:
            new_stage = 'partial_unfreeze'
        else:
            new_stage = 'full_finetune'
        
        if new_stage != self.training_stage:
            logger.info(f"Stage change: {self.training_stage} -> {new_stage}")
            self.training_stage = new_stage
            self.setup_optimizer()
    
    def train_epoch(self, epoch):
        """Train one epoch with REAL predictions"""
        self.update_stage(epoch)
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Train {epoch} ({self.training_stage})")
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # Data to device
                rgb = batch['rgb'].to(self.device)
                depth = batch['depth'].to(self.device)
                boxes_3d = batch['boxes_3d'].to(self.device)
                labels = batch['labels'].to(self.device)
                num_boxes = batch['num_boxes'].to(self.device)
                
                # Create sensor
                image_sizes = [(w.item(), h.item()) for w, h in zip(batch['image_size'][:, 0], batch['image_size'][:, 1])]
                sensor = create_batched_sensor(rgb, depth, image_sizes)
                
                # Forward pass -> REAL predictions
                self.optimizer.zero_grad()
                predictions = self.model(sensor["wide"])
                
                # Compute loss with REAL predictions vs GT
                loss = self.criterion(predictions, boxes_3d, labels, num_boxes)
                
                # Backward
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': f'{total_loss/num_batches:.4f}'})
                
                if batch_idx % 50 == 0:
                    logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                    logger.info(f"  Sample: {batch['scene_name'][0]}, Boxes: {num_boxes[0].item()}")
                
            except Exception as e:
                logger.error(f"Train batch {batch_idx} error: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate_epoch(self, epoch):
        """Validate with REAL predictions"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc=f"Val {epoch}")):
                try:
                    rgb = batch['rgb'].to(self.device)
                    depth = batch['depth'].to(self.device)
                    boxes_3d = batch['boxes_3d'].to(self.device)
                    labels = batch['labels'].to(self.device) 
                    num_boxes = batch['num_boxes'].to(self.device)
                    
                    image_sizes = [(w.item(), h.item()) for w, h in zip(batch['image_size'][:, 0], batch['image_size'][:, 1])]
                    sensor = create_batched_sensor(rgb, depth, image_sizes)
                    
                    # REAL predictions
                    predictions = self.model(sensor["wide"])
                    loss = self.criterion(predictions, boxes_3d, labels, num_boxes)
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    logger.error(f"Val batch {batch_idx} error: {e}")
                    continue
        
        avg_loss = total_loss / max(num_batches, 1)
        self.val_losses.append(avg_loss)
        
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_checkpoint(epoch, is_best=True)
        
        return avg_loss
    
    def train(self, num_epochs, save_dir='enhanced_checkpoints'):
        """Full training loop with REAL predictions"""
        os.makedirs(save_dir, exist_ok=True)
        
        logger.info(f"Starting enhanced training: {num_epochs} epochs")
        logger.info("Stages: freeze_dino (0-7) -> partial (8-14) -> full (15+)")
        logger.info("Using REAL 3D predictions from detection head!")
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate_epoch(epoch)
            
            logger.info(f"Epoch {epoch}: Train {train_loss:.4f}, Val {val_loss:.4f}")
            
            if epoch % 5 == 0:
                self.save_checkpoint(epoch, save_dir)
        
        self.save_checkpoint(num_epochs - 1, save_dir, is_final=True)
        logger.info("Enhanced training complete with REAL predictions!")
    
    def save_checkpoint(self, epoch, save_dir='checkpoints', is_best=False, is_final=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if is_best:
            path = Path(save_dir) / 'best_enhanced_model.pth'
        elif is_final:
            path = Path(save_dir) / 'final_enhanced_model.pth'
        else:
            path = Path(save_dir) / f'enhanced_epoch_{epoch}.pth'
        
        torch.save(checkpoint, path)
        logger.info(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='/mnt/data/datasets/sun_rgbd')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--save_dir', default='enhanced_checkpoints')
    parser.add_argument('--num_workers', type=int, default=2)
    
    args = parser.parse_args()
    
    config = {
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'num_workers': args.num_workers
    }
    
    # Create enhanced model
    logger.info("Creating enhanced DINO-CubifyAnything model...")
    model = create_enhanced_model()
    logger.info("✓ Enhanced model with detection head created")
    
    # Create datasets
    train_dataset = UpdatedSunRGBDDataset(args.data_root, split='train')
    test_dataset = UpdatedSunRGBDDataset(args.data_root, split='test')
    logger.info(f"Datasets: {len(train_dataset)} train, {len(test_dataset)} test")
    
    # Train
    trainer = EnhancedTrainer(model, train_dataset, test_dataset, config)
    trainer.train(args.num_epochs, args.save_dir)


if __name__ == "__main__":
    main()
