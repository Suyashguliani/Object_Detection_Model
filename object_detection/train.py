import torch
import torch.nn as nn  # Add this line with other imports
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from detection_model import DetectionModel
from dataset import VOCDataset, get_transform
from loss import StableYOLOLoss
import torchvision.models as models
import os
import time
import numpy as np
from tqdm import tqdm
import signal


class TimeoutException(Exception):
    pass


def handler(signum, frame):
    raise TimeoutException()


def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU Memory - Allocated: {torch.cuda.memory_allocated() / 1e6:.2f}MB, "
              f"Cached: {torch.cuda.memory_reserved() / 1e6:.2f}MB")


def save_model(model, path, epoch=None, loss=None, config=None):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
        'config': config
    }, path)


# ================= Configuration =================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(4)
torch.backends.quantized.engine = 'qnnpack'

NUM_CLASSES = 20
S = 14
B = 2
EPOCHS = 20
BATCH_SIZE = 8
LR = 1e-5
WARMUP_EPOCHS = 3
MAX_GRAD_NORM = 1.0

# ================= Setup =================
print(f"\n{'=' * 40}")
print(f"{'Training Setup':^40}")
print(f"{'=' * 40}")
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"\n{' Configuration ':-^40}")
print(f"Classes: {NUM_CLASSES} | Grid: {S}x{S} | Boxes: {B}")
print(f"Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | LR: {LR}")

# ================= Model =================
print(f"\n{' Model ':-^40}")
resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
backbone = torch.nn.Sequential(*list(resnet50.children())[:-2])
model = DetectionModel(backbone=backbone, num_classes=NUM_CLASSES, S=S, B=B).to(device)

# Initialize final layer carefully
for m in model.modules():
    if isinstance(m, nn.Conv2d) and m.weight.shape[0] == (B * 5 + NUM_CLASSES):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        nn.init.constant_(m.bias, 0)

print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ================= Loss & Optimizer =================
criterion = StableYOLOLoss(S=S, B=B, C=NUM_CLASSES)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=LR / 100)
scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')

# ================= Data =================
print(f"\n{' Data ':-^40}")
dataset = VOCDataset(root='./data', year='2007', image_set='train',
                     transform=get_transform(), S=S, B=B, C=NUM_CLASSES)

# Data validation
print("Validating dataset...")
for i in range(min(5, len(dataset))):
    img, target = dataset[i]
    if torch.isnan(img).any() or torch.isnan(target).any():
        print(f"NaN detected in sample {i}")
    if (target[..., 25:] < 0).any() or (target[..., 25:] > 1).any():
        print(f"Invalid bbox in sample {i}")

debug_mode = True
if debug_mode:
    dataset = torch.utils.data.Subset(dataset, indices=range(200))
    print("DEBUG: Using 200 sample subset")

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=2 if not debug_mode else 0)

# ================= Training =================
print(f"\n{' Training ':-^40}")
model.train()
best_loss = float('inf')
config = {'S': S, 'B': B, 'C': NUM_CLASSES}

signal.signal(signal.SIGALRM, handler)

for epoch in range(EPOCHS):
    epoch_loss = 0
    valid_batches = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

    # Warmup
    if epoch < WARMUP_EPOCHS:
        lr_scale = (epoch + 1) / WARMUP_EPOCHS
        for param_group in optimizer.param_groups:
            param_group['lr'] = LR * lr_scale

    for batch_idx, (imgs, targets) in enumerate(progress_bar):
        signal.alarm(60)  # Set 60 second timeout
        try:
            # Move data to device
            imgs = imgs.to(device)
            targets = targets.to(device)

            # Forward pass with mixed precision
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                outputs = model(imgs)
                loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Update learning rate
            if epoch >= WARMUP_EPOCHS:
                scheduler.step()

            # Tracking
            epoch_loss += loss.item()
            valid_batches += 1

            progress_bar.set_postfix({
                'loss': loss.item(),
                'lr': optimizer.param_groups[0]['lr']
            })

            print_gpu_memory()

        except Exception as e:
            print(f"\nBatch {batch_idx} error: {str(e)}")
            optimizer.zero_grad()
        finally:
            signal.alarm(0)  # Disable timeout

    # Epoch summary
    if valid_batches > 0:
        avg_loss = epoch_loss / valid_batches
        print(f"\nEpoch {epoch + 1} | Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model, 'best_model.pth', epoch + 1, best_loss, config)
            print(f"New best model saved (loss: {best_loss:.4f})")

        if (epoch + 1) % 5 == 0:
            save_model(model, f'epoch_{epoch + 1}.pth', epoch + 1, avg_loss, config)

# Final save
save_model(model, 'final_model.pth', EPOCHS, best_loss, config)
print("\nTraining complete!")
print(f"Best loss: {best_loss:.4f}")
print(f"Models saved: best_model.pth, final_model.pth")