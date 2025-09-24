#!/bin/bash

# Полный пайплайн DINO-CubifyAnything training
# Usage: bash run_pipeline.sh

set -e  # Exit on error

echo "🚀 DINO-CubifyAnything Training Pipeline"
echo "========================================"

# Directories
PROJECT_DIR="/mnt/data/dino_cubify_integration"
DATA_DIR="/mnt/data/datasets/sun_rgbd"
ADAPTER_DIR="$PROJECT_DIR/integration/adapters"

# Activate conda environment
echo "🔧 Activating environment..."
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate dino_cutr

# Set Python path
export PYTHONPATH="$PROJECT_DIR/ml-cubifyanything:$PROJECT_DIR/MoGe:$ADAPTER_DIR:$PYTHONPATH"

echo "📂 Working directory: $PROJECT_DIR"
cd "$PROJECT_DIR"

# Step 1: Verify integration
echo ""
echo "1️⃣ Verifying DINO-CubifyAnything integration..."
cd "$ADAPTER_DIR"
python final_integration.py

if [ $? -eq 0 ]; then
    echo "✅ Integration verified successfully!"
else
    echo "❌ Integration verification failed!"
    exit 1
fi

# Step 2: Prepare dataset
echo ""
echo "2️⃣ Preparing SUN RGB-D dataset..."
cd "$PROJECT_DIR"

# Check if data files exist
if [ ! -f "$DATA_DIR/SUNRGBDMeta3DBB_v2.mat" ]; then
    echo "❌ SUN RGB-D metadata not found!"
    echo "Please ensure the following files are downloaded to $DATA_DIR:"
    echo "  - SUNRGBDMeta3DBB_v2.mat"
    echo "  - SUNRGBD.zip (and extracted)"
    echo "  - SUNRGBDtoolbox.zip (and extracted)"
    exit 1
fi

# Extract archives if needed
if [ -f "$DATA_DIR/SUNRGBD.zip" ] && [ ! -d "$DATA_DIR/SUNRGBD" ]; then
    echo "📦 Extracting SUNRGBD.zip..."
    cd "$DATA_DIR"
    unzip -q SUNRGBD.zip
    cd "$PROJECT_DIR"
fi

if [ -f "$DATA_DIR/SUNRGBDtoolbox.zip" ] && [ ! -d "$DATA_DIR/SUNRGBDtoolbox" ]; then
    echo "📦 Extracting SUNRGBDtoolbox.zip..."
    cd "$DATA_DIR"
    unzip -q SUNRGBDtoolbox.zip
    cd "$PROJECT_DIR"
fi

# Run data preparation
python prepare_sun_rgbd.py

if [ $? -eq 0 ]; then
    echo "✅ Dataset prepared successfully!"
else
    echo "❌ Dataset preparation failed!"
    exit 1
fi

# Step 3: Check GPU availability
echo ""
echo "3️⃣ Checking GPU availability..."
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB')
else:
    print('❌ No GPU available for training')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ GPU check failed!"
    exit 1
fi

# Step 4: Start training
echo ""
echo "4️⃣ Starting DINO-CubifyAnything training..."
echo "📊 Training will proceed in 3 stages:"
echo "   Stage 1 (5 epochs): Train adapters only"
echo "   Stage 2 (10 epochs): Unfreeze DINO late layers"  
echo "   Stage 3 (15 epochs): Full fine-tuning"
echo ""

# Run training with logging
python train_dino_cubify.py 2>&1 | tee training.log

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Training completed successfully!"
    echo ""
    echo "📁 Generated files:"
    echo "   - best_model.pth: Best model checkpoint"
    echo "   - checkpoint_stage1.pth: Stage 1 checkpoint"
    echo "   - checkpoint_stage2.pth: Stage 2 checkpoint" 
    echo "   - checkpoint_stage3.pth: Stage 3 checkpoint"
    echo "   - training.log: Training logs"
    echo ""
    echo "🔬 Next steps:"
    echo "   1. Evaluate on test set: python evaluate_metrics.py"
    echo "   2. Compare with baseline CuTR results"
    echo "   3. Analyze feature maps: python analyze_features.py"
else
    echo "❌ Training failed! Check training.log for details"
    exit 1
fi

# Step 5: Quick evaluation (optional)
echo ""
read -p "5️⃣ Run quick evaluation on test set? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🔍 Running evaluation..."
    python evaluate_metrics.py 2>&1 | tee evaluation.log
    
    if [ $? -eq 0 ]; then
        echo "✅ Evaluation completed! Results saved to evaluation.log"
    else
        echo "❌ Evaluation failed"
    fi
fi

echo ""
echo "🎯 Pipeline completed!"
echo "📋 Summary:"
echo "   ✅ Integration verified"
echo "   ✅ Dataset prepared"
echo "   ✅ Training completed"
echo "   💾 Model saved to: $PROJECT_DIR/best_model.pth"
echo ""
echo "🚀 Your DINO-CubifyAnything model is ready for production!"
