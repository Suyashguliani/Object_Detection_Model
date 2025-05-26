import torch
import torch.nn as nn
import torchvision.models as models

class DetectionModel(nn.Module):
    def __init__(self, backbone=None, num_classes=20, S=14, B=2):
        super(DetectionModel, self).__init__()
        self.S = S
        self.B = B
        self.C = num_classes

        if backbone is None:
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        else:
            self.backbone = backbone

        self.head = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, self.B * 5 + self.C, kernel_size=1)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        x = x.permute(0, 2, 3, 1)
        return x

if __name__ == "__main__":
    model = DetectionModel(S=14)
    model.eval()
    x = torch.randn(8, 3, 448, 448)
    with torch.no_grad():
        out = model(x)
    print(f"Output shape with 448x448 input: {out.shape}")

    model_224 = DetectionModel(S=7)
    model_224.eval()
    x_224 = torch.randn(8, 3, 224, 224)
    with torch.no_grad():
        out_224 = model_224(x_224)
    print(f"Output shape with 224x224 input: {out_224.shape}")