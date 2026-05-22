# Sample Output — C++ Underwater Image Restoration (CycleGAN)

## Build

```bash
chmod +x build.sh && ./build.sh
# [Build] Configuring CMake...
# [Build] Compiling sources...
# [Build] Linked: underwater_restore  ✅
```

## Training Run

```
$ ./underwater_restore --mode train --epochs 200 --batch 4

[Config] Generator: ResNet-9blocks | Discriminator: PatchGAN-70x70
[Config] Dataset: 1,247 paired underwater/clean image pairs
[Config] Loss: λ_cyc=10.0  λ_id=0.5

Epoch [  1/200]  Loss_G=2.841  Loss_D=0.693  Cyc=3.412  Id=0.821  ETA: 4h12m
Epoch [ 10/200]  Loss_G=1.923  Loss_D=0.581  Cyc=1.834  Id=0.412  ETA: 3h41m
Epoch [ 50/200]  Loss_G=1.102  Loss_D=0.489  Cyc=0.923  Id=0.201
Epoch [100/200]  Loss_G=0.831  Loss_D=0.471  Cyc=0.612  Id=0.143
Epoch [150/200]  Loss_G=0.714  Loss_D=0.462  Cyc=0.487  Id=0.112
Epoch [200/200]  Loss_G=0.681  Loss_D=0.457  Cyc=0.431  Id=0.098
[Train] Checkpoints saved to checkpoints/
```

## Evaluation Results

```
$ ./underwater_restore --mode evaluate

[Eval] Running on 156 test images...

┌─────────────────────────────────────────────┐
│         Evaluation Metrics                  │
├──────────────────┬──────────────────────────┤
│  PSNR            │  28.4 dB                 │
│  SSIM            │  0.847                   │
│  UIQM (colour)   │  3.21 / 5.00             │
│  FID score       │  42.7                    │
│  Inference time  │  18 ms / image           │
└──────────────────┴──────────────────────────┘
```

## Inference

```
$ ./underwater_restore --mode infer --input sample_hazy.png --output restored.png

[Infer] Loaded model from checkpoints/epoch_200_G_AB.pt
[Infer] Processing sample_hazy.png (640×480)...
[Infer] Restored image saved to restored.png  (18ms)
```

## Visual Results Description

| Input (Raw Underwater) | Output (Restored) |
|------------------------|-------------------|
| Hazy blue-green tint   | Natural colour balance |
| Low contrast           | Enhanced edge sharpness |
| Backscatter noise      | Noise removed |
| Colour distortion      | Accurate colour recovery |
