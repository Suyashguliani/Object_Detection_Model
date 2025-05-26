import torch
from torch.utils.data import DataLoader
from detection_model import DetectionModel
from dataset import VOCDataset, get_transform
from utils.metrics import calculate_map, non_max_suppression, _convert_outputs_to_bboxes, _convert_targets_to_bboxes
import torchvision.models as models
import os
from tqdm import tqdm

# Define common parameters
NUM_CLASSES = 20
S = 14
B = 2
BATCH_SIZE = 8

# Define class names
CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

model_path = os.path.join(os.path.dirname(__file__), 'model.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load ResNet50 backbone
resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
backbone = torch.nn.Sequential(*list(resnet50.children())[:-2])

# Initialize model and load weights
model = DetectionModel(backbone=backbone, num_classes=NUM_CLASSES, S=S, B=B).to(device)
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded from {model_path}")
except FileNotFoundError:
    print(f"Error: model.pth not found at {model_path}. Please train the model first.")
    exit()

model.eval()

# Prepare validation dataset and dataloader
dataset = VOCDataset(root='./data', year='2007', image_set='val', transform=get_transform(), S=S, B=B, C=NUM_CLASSES)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

all_pred_boxes = []
all_true_boxes = []

print("Starting evaluation...")
with torch.no_grad():
    for imgs, targets in tqdm(dataloader, desc="Evaluating"):
        imgs = imgs.to(device)
        outputs = model(imgs)

        # Process outputs and targets
        batch_pred_boxes = _convert_outputs_to_bboxes(outputs, S, B, NUM_CLASSES, threshold=0.01)
        batch_true_boxes = _convert_targets_to_bboxes(targets, S, B, NUM_CLASSES)

        # Apply NMS per image
        final_boxes = non_max_suppression(batch_pred_boxes, iou_threshold=0.45, conf_threshold=0.3)

        all_pred_boxes.extend(final_boxes)
        all_true_boxes.extend(batch_true_boxes)

# Calculate mAP
mAP_val = calculate_map(all_pred_boxes, all_true_boxes, iou_threshold=0.5, class_names=CLASS_NAMES)

print(f"Evaluation done. mAP (IoU=0.5): {mAP_val:.4f}")