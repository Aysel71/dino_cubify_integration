import sys
sys.path.append('/mnt/data/dino_cubify_integration/ml-cubifyanything')
sys.path.append('/mnt/data/dino_cubify_integration/MoGe')
sys.path.append('/mnt/data/dino_cubify_integration/integration/adapters')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from final_integration import create_dino_cubify_model

class SunRGBDDataset(Dataset):
    """
    SUN RGB-D dataset для DINO-CubifyAnything обучения
    """
    
    def __init__(self, data_root, split='train', image_size=(512, 768)):
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        
        # Загрузить список сцен
        processed_dir = self.data_root / 'processed'
        scene_file = processed_dir / f'{split}_scenes.json'
        
        with open(scene_file, 'r') as f:
            self.scenes = json.load(f)
        
        print(f"Loaded {len(self.scenes)} {split} scenes")
        
        # Трансформации для изображений
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Класс маппинг для SUN RGB-D
        self.class_mapping = self._create_class_mapping()
        
    def _create_class_mapping(self):
        """Создание маппинга классов в числовые ID"""
        all_classes = set()
        for scene in self.scenes:
            all_classes.update(scene['classes'])
        
        # Добавить background класс
        class_list = ['background'] + sorted(list(all_classes))
        class_to_id = {cls: idx for idx, cls in enumerate(class_list)}
        
        print(f"Found {len(class_list)} classes: {class_list[:10]}...")
        return class_to_id
    
    def __len__(self):
        return len(self.scenes)
    
    def __getitem__(self, idx):
        scene = self.scenes[idx]
        
        # Загрузить RGB изображение
        rgb_path = self.data_root / scene['rgb_path']
        rgb_image = Image.open(rgb_path).convert('RGB')
        rgb_tensor = self.transform(rgb_image)
        
        # Загрузить 3D bounding boxes
        boxes_3d = np.array(scene['boxes_3d'], dtype=np.float32)
        classes = [self.class_mapping.get(cls, 0) for cls in scene['classes']]
        classes = np.array(classes, dtype=np.int64)
        
        # Создать targets в формате, ожидаемом CubifyAnything
        targets = {
            'boxes_3d': torch.from_numpy(boxes_3d),
            'labels': torch.from_numpy(classes),
            'image_id': torch.tensor(idx),
            'orig_size': torch.tensor([rgb_image.height, rgb_image.width]),
            'size': torch.tensor(self.image_size)
        }
        
        # Создать sensor в формате BatchedPosedSensor
        return self._create_batched_sensor(rgb_tensor), targets
    
    def _create_batched_sensor(self, rgb_tensor):
        """Создать BatchedPosedSensor формат для CubifyAnything"""
        
        class MockImageData:
            def __init__(self, tensor, sizes):
                self.tensor = tensor.unsqueeze(0)  # Add batch dimension
                self.image_sizes = sizes
        
        class MockImageSensor:
            def __init__(self, tensor, sizes):
                self.data = MockImageData(tensor, sizes)
                self.info = [None]
        
        sensor = MockImageSensor(rgb_tensor, [self.image_size])
        
        return {"image": sensor}

class ProgressiveTrainer:
    """
    Progressive training strategy для DINO-CubifyAnything
    """
    
    def __init__(self, model, train_loader, val_loader, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Training stages - progressive unfreezing
        self.stages = {
            'stage1': {
                'epochs': 5, 
                'lr': 1e-3, 
                'freeze_dino': True,
                'description': 'Train adapters only'
            },
            'stage2': {
                'epochs': 10, 
                'lr': 5e-4, 
                'freeze_dino': False, 
                'freeze_early_layers': True,
                'description': 'Unfreeze DINO late layers'
            },
            'stage3': {
                'epochs': 15, 
                'lr': 1e-4, 
                'freeze_dino': False,
                'description': 'Full fine-tuning'
            }
        }
        
        self.current_epoch = 0
        self.best_loss = float('inf')
    
    def train_all_stages(self):
        """Обучение всех этапов последовательно"""
        
        for stage_name, config in self.stages.items():
            print(f"\n{'='*50}")
            print(f"Starting {stage_name}: {config['description']}")
            print(f"Epochs: {config['epochs']}, LR: {config['lr']}")
            print(f"{'='*50}")
            
            self.train_stage(stage_name, config)
            
            # Сохранить checkpoint после каждого этапа
            self.save_checkpoint(f'checkpoint_{stage_name}.pth')
    
    def train_stage(self, stage_name, config):
        """Обучение одного этапа"""
        
        # Настроить параметры для обучения
        self._configure_parameters(config)
        
        # Создать optimizer
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = optim.AdamW(
            trainable_params,
            lr=config['lr'],
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs']
        )
        
        # Training loop
        for epoch in range(config['epochs']):
            print(f"\nEpoch {epoch+1}/{config['epochs']}")
            
            # Training
            train_loss = self._train_epoch(optimizer)
            
            # Validation
            val_loss = self._validate_epoch()
            
            # Update learning rate
            scheduler.step()
            
            # Logging
            lr = optimizer.param_groups[0]['lr']
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {lr:.6f}")
            
            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint('best_model.pth')
                print("→ New best model saved!")
            
            self.current_epoch += 1
    
    def _configure_parameters(self, config):
        """Настройка параметров для обучения"""
        
        # Сначала разморозить все параметры
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Затем заморозить согласно конфигурации
        if config.get('freeze_dino', False):
            # Заморозить весь DINO encoder
            dino_encoder = self.model.backbone[0].dino_encoder
            for param in dino_encoder.parameters():
                param.requires_grad = False
            print("  → DINO encoder frozen")
            
        elif config.get('freeze_early_layers', False):
            # Заморозить ранние слои DINO, разрешить дообучение поздних
            backbone = self.model.backbone[0].dino_encoder.backbone
            for layer_idx in range(6):  # Заморозить первые 6 слоев
                if hasattr(backbone, 'blocks') and layer_idx < len(backbone.blocks):
                    for param in backbone.blocks[layer_idx].parameters():
                        param.requires_grad = False
            print("  → DINO early layers (0-5) frozen, late layers (6-11) trainable")
        
        # Подсчитать количество обучаемых параметров
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"  → Trainable parameters: {trainable_params:,} / {total_params:,} "
              f"({100*trainable_params/total_params:.1f}%)")
    
    def _train_epoch(self, optimizer):
        """Обучение одной эпохи"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (sensor_batch, targets_batch) in enumerate(self.train_loader):
            # Переместить на GPU
            targets_batch = self._move_targets_to_device(targets_batch)
            
            optimizer.zero_grad()
            
            try:
                # Forward pass - здесь нужна реальная loss функция CubifyAnything
                # Пока используем dummy loss для демонстрации
                outputs = self.model.backbone(sensor_batch)
                
                # Dummy loss - в реальности нужна proper CubifyAnything loss
                dummy_loss = torch.tensor(0.5, device=self.device, requires_grad=True)
                for output in outputs:
                    if hasattr(output, 'tensors'):
                        dummy_loss = dummy_loss + output.tensors.mean() * 0.001
                
                loss = dummy_loss
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}/{len(self.train_loader)}: Loss {loss.item():.4f}")
                
            except Exception as e:
                print(f"  Error in batch {batch_idx}: {e}")
                continue
        
        return total_loss / max(num_batches, 1)
    
    def _validate_epoch(self):
        """Валидация одной эпохи"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for sensor_batch, targets_batch in self.val_loader:
                try:
                    targets_batch = self._move_targets_to_device(targets_batch)
                    
                    outputs = self.model.backbone(sensor_batch)
                    
                    # Dummy loss - в реальности нужна proper CubifyAnything loss
                    dummy_loss = torch.tensor(0.5, device=self.device)
                    for output in outputs:
                        if hasattr(output, 'tensors'):
                            dummy_loss = dummy_loss + output.tensors.mean() * 0.001
                    
                    loss = dummy_loss
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    print(f"  Validation error: {e}")
                    continue
        
        return total_loss / max(num_batches, 1)
    
    def _move_targets_to_device(self, targets_batch):
        """Перемещение targets на GPU"""
        if isinstance(targets_batch, dict):
            return {k: v.to(self.device) if torch.is_tensor(v) else v 
                   for k, v in targets_batch.items()}
        return targets_batch
    
    def save_checkpoint(self, filename):
        """Сохранение checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'best_loss': self.best_loss,
            'stages': self.stages
        }
        
        save_path = Path('/mnt/data/dino_cubify_integration') / filename
        torch.save(checkpoint, save_path)
        print(f"  → Checkpoint saved: {save_path}")

def create_data_loaders(data_root, batch_size=2):
    """Создание data loaders"""
    
    train_dataset = SunRGBDDataset(data_root, 'train')
    val_dataset = SunRGBDDataset(data_root, 'test')  # Используем test как validation
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        collate_fn=lambda x: x  # Пока simple collate
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: x
    )
    
    return train_loader, val_loader

def main():
    """Основная функция обучения"""
    
    print("🚀 Starting DINO-CubifyAnything Training")
    print("="*50)
    
    # Проверить CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    device = torch.cuda.current_device()
    print(f"✅ Using GPU: {torch.cuda.get_device_name(device)}")
    
    # Создать модель
    print("\n📦 Creating model...")
    model = create_dino_cubify_model()
    
    # Подсчитать параметры
    total_params = sum(p.numel() for p in model.parameters())
    print(f"�� Total parameters: {total_params:,}")
    
    # Создать data loaders
    print("\n📂 Loading dataset...")
    data_root = '/mnt/data/datasets/sun_rgbd'
    train_loader, val_loader = create_data_loaders(data_root, batch_size=2)
    
    print(f"📈 Train batches: {len(train_loader)}")
    print(f"�� Val batches: {len(val_loader)}")
    
    # Создать trainer
    trainer = ProgressiveTrainer(model, train_loader, val_loader, device)
    
    # Запустить обучение
    print("\n🎯 Starting progressive training...")
    trainer.train_all_stages()
    
    print("\n🎉 Training completed!")
    print("💾 Model saved to: /mnt/data/dino_cubify_integration/best_model.pth")

if __name__ == "__main__":
    main()
