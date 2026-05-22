# Underwater Image Restoration - Training Configuration Guide

## Overview

This guide provides detailed hyperparameter configurations for different training scenarios using the CycleGAN architecture.

---

## Default Configuration (Balanced)

**Best for**: General underwater image restoration with EUVP dataset

```cpp
// include/config.hpp

// IMAGE PROCESSING
constexpr int IMAGE_HEIGHT = 256;
constexpr int IMAGE_WIDTH = 256;
constexpr int IMAGE_CHANNELS = 3;

// TRAINING HYPERPARAMETERS
constexpr int BATCH_SIZE = 4;
constexpr int NUM_EPOCHS = 200;
constexpr int SAVE_INTERVAL = 10;
constexpr float LEARNING_RATE = 0.0002f;
constexpr float BETA1 = 0.5f;
constexpr float BETA2 = 0.999f;

// LOSS WEIGHTS
constexpr float LAMBDA_CYCLE = 10.0f;      // Primary constraint
constexpr float LAMBDA_IDENTITY = 5.0f;    // Color preservation
constexpr float LAMBDA_ADV = 1.0f;         // Adversarial strength

// ARCHITECTURE
constexpr int GENERATOR_FILTERS = 64;
constexpr int DISCRIMINATOR_FILTERS = 64;

// DEVICE
const std::string DEVICE = "cuda";
```

**Expected Results**:
- Training time: ~8-12 hours on RTX 3090
- Convergence: ~100 epochs
- Generator loss: 0.3-0.5
- Final restoration quality: Good

---

## Fast Training Configuration (Lower Resources)

**Best for**: Quick prototyping, limited VRAM

```cpp
// REDUCED BATCH SIZE
constexpr int BATCH_SIZE = 2;              // ↓ from 4

// SMALLER IMAGE RESOLUTION
constexpr int IMAGE_HEIGHT = 128;          // ↓ from 256
constexpr int IMAGE_WIDTH = 128;           // ↓ from 256

// FEWER EPOCHS
constexpr int NUM_EPOCHS = 100;            // ↓ from 200
constexpr int SAVE_INTERVAL = 5;           // ↓ from 10

// REDUCED MODEL SIZE
constexpr int GENERATOR_FILTERS = 32;      // ↓ from 64
constexpr int DISCRIMINATOR_FILTERS = 32;  // ↓ from 64
```

**Benefits**:
- 3-4× faster training
- Lower VRAM: ~2 GB (vs 6 GB)
- Suitable for GTX 1660 or better

**Trade-offs**:
- Lower output quality
- May need more epochs to converge
- Smaller receptive field

---

## High-Quality Configuration (High Resources)

**Best for**: Production-grade restoration, unlimited compute

```cpp
// LARGER BATCH SIZE
constexpr int BATCH_SIZE = 8;              // ↑ from 4

// HIGHER IMAGE RESOLUTION
constexpr int IMAGE_HEIGHT = 512;          // ↑ from 256
constexpr int IMAGE_WIDTH = 512;           // ↑ from 256

// MORE TRAINING
constexpr int NUM_EPOCHS = 300;            // ↑ from 200
constexpr int SAVE_INTERVAL = 5;           // More checkpoints

// LARGER MODEL
constexpr int GENERATOR_FILTERS = 128;     // ↑ from 64
constexpr int DISCRIMINATOR_FILTERS = 128; // ↑ from 64

// OPTIMIZED LEARNING RATE
constexpr float LEARNING_RATE = 0.0001f;   // ↓ for stability
```

**Requirements**:
- VRAM: 12-16 GB (RTX 3080, RTX 4090)
- Training time: 24-36 hours
- Disk space: 50+ GB for checkpoints

**Benefits**:
- Superior restoration quality
- Better fine details preservation
- Enhanced color accuracy

---

## Color-Preservation Configuration

**Best for**: Maintaining natural underwater colors while removing haze

```cpp
// INCREASE COLOR CONSISTENCY
constexpr float LAMBDA_IDENTITY = 15.0f;   // ↑ from 5.0 (3× increase)
constexpr float LAMBDA_CYCLE = 10.0f;      // Keep moderate

// MODERATE ADVERSARIAL
constexpr float LAMBDA_ADV = 0.5f;         // ↓ from 1.0

// HIGHER LEARNING RATE FOR FASTER ADAPTATION
constexpr float LEARNING_RATE = 0.0003f;   // ↑ from 0.0002

// STANDARD SIZE
constexpr int IMAGE_HEIGHT = 256;
constexpr int IMAGE_WIDTH = 256;
constexpr int BATCH_SIZE = 4;
```

**Use Case**: Coral reef imaging, species identification

**Expected Output**: Less desaturated, warmer tones

---

## Detail-Enhancement Configuration

**Best for**: Fine structure recovery, edge preservation

```cpp
// REDUCE IDENTITY LOSS (allow more modification)
constexpr float LAMBDA_IDENTITY = 2.0f;    // ↓ from 5.0

// INCREASE CYCLE CONSISTENCY
constexpr float LAMBDA_CYCLE = 15.0f;      // ↑ from 10.0

// STRONG ADVERSARIAL COMPONENT
constexpr float LAMBDA_ADV = 2.0f;         // ↑ from 1.0

// MORE ITERATIONS
constexpr int NUM_EPOCHS = 250;
constexpr float LEARNING_RATE = 0.00015f;
```

**Use Case**: Wreck inspection, geological surveys

**Expected Output**: Sharp edges, enhanced contrast, recovered details

---

## Memory-Efficient Configuration (GTX 1080 / RTX 2070)

**Best for**: Mid-range GPUs with 8 GB VRAM

```cpp
constexpr int BATCH_SIZE = 2;
constexpr int IMAGE_HEIGHT = 256;
constexpr int IMAGE_WIDTH = 256;
constexpr int NUM_EPOCHS = 150;

constexpr int GENERATOR_FILTERS = 64;
constexpr int DISCRIMINATOR_FILTERS = 64;

constexpr float LEARNING_RATE = 0.0002f;
constexpr float LAMBDA_CYCLE = 10.0f;
constexpr float LAMBDA_IDENTITY = 5.0f;

const std::string DEVICE = "cuda";
```

**Tips for Memory Management**:

1. **Gradient Accumulation** (modify training loop):
```cpp
// Accumulate gradients over 2 batches
for (int acc_step = 0; acc_step < 2; ++acc_step) {
    // Process batch
    // Backward without optimizer step
    loss.backward();
}
optimizer.step();  // Update after accumulation
```

2. **Reduce Checkpoint Frequency**:
```cpp
constexpr int SAVE_INTERVAL = 20;  // Save every 20 epochs
```

3. **Enable Mixed Precision** (requires PyTorch 1.6+):
```cpp
// Use Float16 for forward pass, Float32 for loss
auto scale_factor = 65536.0f;
loss = loss * scale_factor;
loss.backward();
optimizer.step();
loss = loss / scale_factor;
```

---

## Curriculum Learning Configuration

**Best for**: Progressive training from easy to hard samples

```cpp
// Phase 1: Warm-up (low adversarial weight)
// Epochs 0-50
constexpr float LAMBDA_CYCLE = 10.0f;
constexpr float LAMBDA_ADV = 0.1f;

// Phase 2: Main training (balanced)
// Epochs 50-150
constexpr float LAMBDA_CYCLE = 10.0f;
constexpr float LAMBDA_ADV = 1.0f;

// Phase 3: Fine-tuning (high adversarial)
// Epochs 150+
constexpr float LAMBDA_CYCLE = 5.0f;
constexpr float LAMBDA_ADV = 2.0f;
```

**Implementation in training loop**:
```cpp
for (int epoch = 1; epoch <= NUM_EPOCHS; ++epoch) {
    float current_lambda_adv = LAMBDA_ADV;

    if (epoch < 50) {
        current_lambda_adv = 0.1f;
    } else if (epoch < 150) {
        current_lambda_adv = 1.0f;
    } else {
        current_lambda_adv = 2.0f;
    }

    // Use current_lambda_adv in loss computation
}
```

---

## Loss Weight Tuning Guide

### Understanding Loss Components

| Loss Type | Weight | Effect | Range |
|-----------|--------|--------|-------|
| **Cycle Consistency** | LAMBDA_CYCLE | Forces A→B→A reconstruction | 5-20 |
| **Identity** | LAMBDA_IDENTITY | Preserves colors for in-domain images | 2-15 |
| **Adversarial** | LAMBDA_ADV | Realism vs preservation balance | 0.5-5 |

### Tuning Strategy

**Too much haze remaining?**
- ↑ Increase `LAMBDA_ADV` (stronger GAN)
- ↓ Decrease `LAMBDA_IDENTITY` (allow more changes)

**Colors too desaturated?**
- ↑ Increase `LAMBDA_IDENTITY` (preserve original colors)
- ↓ Decrease `LAMBDA_ADV` (less aggressive generation)

**Artifacts/noise in output?**
- ↑ Increase `LAMBDA_CYCLE` (better reconstruction)
- ↓ Decrease `LAMBDA_ADV` (less adversarial pressure)

**Slow convergence?**
- ↑ Increase `LEARNING_RATE` (but watch for instability)
- ↓ Decrease `LAMBDA_CYCLE` (easier optimization)

---

## GPU Memory Usage by Configuration

| Config | Batch Size | Resolution | Memory (GB) | Training Time (hrs) |
|--------|-----------|-----------|------------|-------------------|
| **Minimal** | 1 | 128×128 | 1.5 | 4-6 |
| **Fast** | 2 | 128×128 | 2.5 | 6-8 |
| **Balanced** | 4 | 256×256 | 6 | 10-12 |
| **High-Quality** | 8 | 512×512 | 14 | 24-36 |

---

## Hyperparameter Search Strategy

### Random Search Template

```python
# Optional: Use Python + PyTorch to generate configs
import random

configs = []
for _ in range(10):
    config = {
        'learning_rate': random.uniform(0.0001, 0.001),
        'lambda_cycle': random.uniform(5, 20),
        'lambda_identity': random.uniform(2, 15),
        'batch_size': random.choice([2, 4, 8]),
    }
    configs.append(config)

# Generate C++ header files for each config
```

### Grid Search (Manual)

```
Learning rates:  0.0001, 0.0002, 0.0005
Lambda cycle:    5, 10, 15, 20
Lambda identity: 2, 5, 10, 15
Lambda adv:      0.5, 1.0, 2.0

Total: 4 × 4 × 4 × 3 = 192 combinations
```

### Validation Strategy

After training, evaluate on test set:

```bash
# Generate metrics
./restore_images data/EUVP/test/murky output_test

# Compare with reference images
# Metrics: PSNR, SSIM (computed in inference)
```

---

## Multi-GPU Training (Advanced)

For systems with multiple GPUs, modify training loop:

```cpp
// Distribute batch across GPUs
torch::Device device0("cuda:0");
torch::Device device1("cuda:1");

auto batch_size_per_gpu = batch_size / 2;

torch::Tensor real_A_gpu0 = real_A.slice(0, 0, batch_size_per_gpu).to(device0);
torch::Tensor real_A_gpu1 = real_A.slice(0, batch_size_per_gpu).to(device1);

// Process on each GPU independently
// Synchronize gradients before optimizer step
```

---

## Monitoring Training Progress

### Recommended Metric Tracking

```
Create plot of:
- Generator loss (A2B, B2A)
- Discriminator loss (A, B)
- Cycle consistency loss
- Identity loss
```

Example monitoring script:
```bash
#!/bin/bash
# Parse training log and plot
grep "^[0-9]" training.log | \
    awk '{print $4, $6, $8, $10, $12}' > losses.txt

# Use gnuplot or Python to visualize
python3 plot_losses.py losses.txt
```

---

## Checkpoint Management

### Best Practice

```bash
# Keep best N checkpoints by validation metric
for epoch in {10,20,30,40,50}; do
    # Evaluate on validation set
    # Keep if better than previous best

    # Archive old checkpoints
    tar -czf checkpoints_archive_epoch_${epoch}.tar.gz \
        checkpoints/generator_*_epoch_${epoch}.pt
done
```

---

## Next Steps

1. **Start with Balanced config** for your dataset
2. **Monitor training** for first 20 epochs
3. **Adjust hyperparameters** if needed
4. **Validate on test set** every 50 epochs
5. **Save best model** based on validation metrics

---

**Last Updated**: April 2026
