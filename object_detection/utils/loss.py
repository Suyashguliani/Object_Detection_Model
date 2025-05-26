import torch
import torch.nn as nn
import torch.nn.functional as F


class StableYOLOLoss(nn.Module):
    def __init__(self, S=14, B=2, C=20, lambda_coord=5.0, lambda_noobj=0.5):
        super(StableYOLOLoss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.epsilon = 1e-7

    def forward(self, predictions, targets):
        """
        predictions shape: (batch_size, S, S, B*5 + C)
        targets shape: (batch_size, S, S, 5 + C)
        """
        batch_size = predictions.size(0)

        # Reshape predictions
        pred_boxes = predictions[..., :self.B * 5].reshape(batch_size, self.S, self.S, self.B, 5)
        pred_class = predictions[..., self.B * 5:]

        # Extract target components
        target_boxes = targets[..., :5]
        target_class = targets[..., 5:]

        # Calculate object presence mask
        obj_mask = target_boxes[..., 4:5] == 1  # (batch_size, S, S, 1)
        noobj_mask = target_boxes[..., 4:5] == 0

        # ===== 1. Localization Loss =====
        # Calculate best IoU for each cell
        ious = []
        for b in range(self.B):
            iou = self._calculate_iou(pred_boxes[..., b, :4], target_boxes[..., :4])
            ious.append(iou)
        ious = torch.stack(ious, dim=-1)  # (batch_size, S, S, B)

        # Get best box for each cell
        best_iou, best_box = torch.max(ious, dim=-1, keepdim=True)  # (batch_size, S, S, 1)

        # Create box mask
        box_mask = torch.zeros_like(ious)
        box_mask.scatter_(-1, best_box.unsqueeze(-1), 1)
        box_mask = box_mask.bool()

        # Calculate coordinate loss only for best boxes
        xy_loss = (pred_boxes[..., :2] - target_boxes[..., :2].unsqueeze(-2)).pow(2)
        wh_loss = (torch.sqrt(pred_boxes[..., 2:4] + self.epsilon) -
                   torch.sqrt(target_boxes[..., 2:4].unsqueeze(-2) + self.epsilon)).pow(2)

        loc_loss = (xy_loss + wh_loss) * box_mask.unsqueeze(-1) * obj_mask.unsqueeze(-1)
        loc_loss = loc_loss.sum()

        # ===== 2. Confidence Loss =====
        pred_conf = torch.sigmoid(pred_boxes[..., 4:5])

        # Object confidence loss
        obj_conf_loss = F.binary_cross_entropy(
            pred_conf,
            target_boxes[..., 4:5].unsqueeze(-2),
            reduction='none'
        ) * obj_mask.unsqueeze(-1)

        # No-object confidence loss
        noobj_conf_loss = F.binary_cross_entropy(
            pred_conf,
            torch.zeros_like(pred_conf),
            reduction='none'
        ) * noobj_mask.unsqueeze(-1) * self.lambda_noobj

        conf_loss = (obj_conf_loss + noobj_conf_loss).sum()

        # ===== 3. Classification Loss =====
        class_loss = F.binary_cross_entropy(
            torch.sigmoid(pred_class),
            target_class,
            reduction='none'
        ).sum(dim=-1) * obj_mask.squeeze(-1)

        class_loss = class_loss.sum()

        # Total loss
        total_loss = (self.lambda_coord * loc_loss +
                      conf_loss +
                      class_loss)

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            raise RuntimeError("Invalid loss value")

        return total_loss

    def _calculate_iou(self, pred_boxes, target_boxes):
        """Calculate IoU between predicted and target boxes"""
        # Convert from (cx, cy, w, h) to (x1, y1, x2, y2)
        pred_boxes = self._convert_to_corners(pred_boxes)
        target_boxes = self._convert_to_corners(target_boxes)

        # Intersection coordinates
        inter_x1 = torch.max(pred_boxes[..., 0], target_boxes[..., 0])
        inter_y1 = torch.max(pred_boxes[..., 1], target_boxes[..., 1])
        inter_x2 = torch.min(pred_boxes[..., 2], target_boxes[..., 2])
        inter_y2 = torch.min(pred_boxes[..., 3], target_boxes[..., 3])

        # Intersection area
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

        # Union area
        pred_area = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])
        target_area = (target_boxes[..., 2] - target_boxes[..., 0]) * (target_boxes[..., 3] - target_boxes[..., 1])
        union_area = pred_area + target_area - inter_area + self.epsilon

        return inter_area / union_area

    def _convert_to_corners(self, boxes):
        """Convert from (cx, cy, w, h) to (x1, y1, x2, y2)"""
        corners = torch.zeros_like(boxes)
        corners[..., 0] = boxes[..., 0] - boxes[..., 2] / 2  # x1
        corners[..., 1] = boxes[..., 1] - boxes[..., 3] / 2  # y1
        corners[..., 2] = boxes[..., 0] + boxes[..., 2] / 2  # x2
        corners[..., 3] = boxes[..., 1] + boxes[..., 3] / 2  # y2
        return corners