import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import os
from detection_model import DetectionModel
from dataset import get_transform
from utils.metrics import non_max_suppression, _convert_outputs_to_bboxes
import torchvision.models as models

# Define common parameters
NUM_CLASSES = 20
S = 14
B = 2

# Define class names
CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# Model and device setup
model_path = os.path.join(os.path.dirname(__file__), 'model.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load backbone and model
resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
backbone = torch.nn.Sequential(*list(resnet50.children())[:-2])
model = DetectionModel(backbone=backbone, num_classes=NUM_CLASSES, S=S, B=B).to(device)

# Load trained weights
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded from {model_path}")
except FileNotFoundError:
    print(f"Error: model.pth not found at {model_path}. Please train the model first.")
    exit()

model.eval()

# Image transformation
transform = get_transform()


def detect_and_visualize(image_path, output_dir="demo_output", conf_threshold=0.3, iou_threshold=0.45):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Processing {image_path}...")
    original_image = Image.open(image_path).convert("RGB")
    img_tensor = transform(original_image).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(img_tensor)

    all_detections_for_image = _convert_outputs_to_bboxes(predictions, S, B, NUM_CLASSES, threshold=0.01)
    final_boxes = non_max_suppression(all_detections_for_image, iou_threshold, conf_threshold)

    draw = ImageDraw.Draw(original_image)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageDraw.Draw(Image.new("RGB", (1, 1))).getfont()

    for box in final_boxes:
        x1, y1, x2, y2, confidence, class_prob, class_idx = box

        img_width, img_height = original_image.size
        x1_abs = int(x1 * img_width)
        y1_abs = int(y1 * img_height)
        x2_abs = int(x2 * img_width)
        y2_abs = int(y2 * img_height)

        label = f"{CLASS_NAMES[class_idx]}: {confidence:.2f}"

        draw.rectangle([x1_abs, y1_abs, x2_abs, y2_abs], outline="red", width=2)
        draw.text((x1_abs + 5, y1_abs + 5), label, fill="red", font=font)

    output_image_path = os.path.join(output_dir, os.path.basename(image_path).replace(".jpg", "_detected.jpg"))
    original_image.save(output_image_path)
    print(f"Detected objects saved to {output_image_path}")


if __name__ == "__main__":
    test_images = [
        "./data/VOCdevkit/VOC2007/JPEGImages/000005.jpg",
    ]

    for img_path in test_images:
        if os.path.exists(img_path):
            detect_and_visualize(img_path)
        else:
            print(f"Warning: Test image not found: {img_path}")
            print("Please ensure VOC 2007 dataset is downloaded and extracted, and update test_images paths.")