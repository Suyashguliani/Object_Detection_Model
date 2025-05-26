import torch
import collections


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.
    box1: (tensor) [x1, y1, x2, y2] or [cx, cy, w, h]
    box2: (tensor) [x1, y1, x2, y2] or [cx, cy, w, h]
    x1y1x2y2: If True, boxes are in (x1, y1, x2, y2) format.
              If False, boxes are in (center_x, center_y, width, height) format.
    """
    if not x1y1x2y2:
        # Convert to (x1, y1, x2, y2)
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0] - box1[..., 2] / 2, box1[..., 1] - box1[..., 3] / 2, \
                                     box1[..., 0] + box1[..., 2] / 2, box1[..., 1] + box1[..., 3] / 2
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0] - box2[..., 2] / 2, box2[..., 1] - box2[..., 3] / 2, \
                                     box2[..., 0] + box2[..., 2] / 2, box2[..., 3] + box2[..., 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]

    # Get the coordinates of the intersection rectangle
    x1_inter = torch.max(b1_x1, b2_x1)
    y1_inter = torch.max(b1_y1, b2_y1)
    x2_inter = torch.min(b1_x2, b2_x2)
    y2_inter = torch.min(b1_y2, b2_y2)

    # Intersection area
    intersection_area = (x2_inter - x1_inter).clamp(0) * (y2_inter - y1_inter).clamp(0)

    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - intersection_area + 1e-16

    iou = intersection_area / union_area

    return iou


def non_max_suppression(bboxes, iou_threshold, conf_threshold):
    """
    Performs Non-Maximum Suppression (NMS) on bounding boxes.
    bboxes: (list) A list of lists/tensors, where each inner list/tensor is
            [x, y, w, h, confidence, class_prob, class_idx]
            or [x1, y1, x2, y2, confidence, class_prob, class_idx]
            It's crucial that the bbox coordinates are in (x1, y1, x2, y2) format
            for this NMS implementation.
    iou_threshold: (float) IoU threshold for suppressing boxes.
    conf_threshold: (float) Confidence threshold for filtering boxes before NMS.
    Returns: (list) A list of selected bounding boxes after NMS.
    """
    # Filter out low-confidence boxes
    bboxes = [box for box in bboxes if box[4] > conf_threshold]
    if not bboxes:
        return []

    # Sort by confidence in descending order
    bboxes = sorted(bboxes, key=lambda x: x[4], reverse=True)

    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes_after_nms.append(chosen_box)

        remaining_boxes = []
        for box in bboxes:
            if bbox_iou(torch.tensor(chosen_box[0:4]), torch.tensor(box[0:4]), x1y1x2y2=True) < iou_threshold \
                    or chosen_box[6] != box[6]:
                remaining_boxes.append(box)
        bboxes = remaining_boxes
    return bboxes_after_nms


def _convert_outputs_to_bboxes(outputs, S, B, C, threshold=0.5, iou_threshold=0.45):
    """
    Converts raw model output (batch, S, S, B*5+C) to a list of detected bounding boxes.
    Each box format: [x1, y1, x2, y2, obj_conf * class_prob, class_index]
    Outputs are relative to image (0-1).
    """
    detections = []
    # outputs shape: (batch_size, S, S, B*5 + C)
    for batch_idx in range(outputs.size(0)):
        output = outputs[batch_idx]  # (S, S, B*5 + C)
        for i in range(S):
            for j in range(S):
                for b in range(B):
                    # Extract box confidence and class probabilities
                    box_data = output[i, j, b * 5: (b + 1) * 5]  # (conf, x, y, w, h)
                    obj_confidence = torch.sigmoid(box_data[0])

                    # Convert box coordinates (relative to cell and image) to (x1, y1, x2, y2) relative to image
                    box_x_offset_in_cell = box_data[1]
                    box_y_offset_in_cell = box_data[2]
                    box_width_rel_img = box_data[3]
                    box_height_rel_img = box_data[4]

                    # Convert cell-relative x,y to image-relative x,y
                    center_x_rel_img = (j + box_x_offset_in_cell) / S
                    center_y_rel_img = (i + box_y_offset_in_cell) / S

                    x1 = center_x_rel_img - (box_width_rel_img / 2)
                    y1 = center_y_rel_img - (box_height_rel_img / 2)
                    x2 = center_x_rel_img + (box_width_rel_img / 2)
                    y2 = center_y_rel_img + (box_height_rel_img / 2)

                    class_probs = torch.softmax(output[i, j, B * 5: B * 5 + C], dim=0)
                    max_class_prob, class_idx = torch.max(class_probs, dim=0)

                    # Final confidence for this detection is obj_confidence * max_class_prob
                    final_confidence = obj_confidence * max_class_prob

                    if final_confidence > threshold:
                        detections.append([
                            x1.item(), y1.item(), x2.item(), y2.item(),
                            final_confidence.item(),
                            max_class_prob.item(),
                            class_idx.item()
                        ])
    return detections


def _convert_targets_to_bboxes(targets, S, B, C):
    """
    Converts ground truth target tensor (batch, S, S, B*5+C) to a list of true bounding boxes.
    Each box format: [x1, y1, x2, y2, class_index] (no confidence needed for GT)
    Coordinates are relative to image (0-1).
    """
    true_boxes = []
    for batch_idx in range(targets.size(0)):
        target = targets[batch_idx]  # (S, S, B*5+C)
        for i in range(S):
            for j in range(S):
                obj_conf_gt = target[i, j, C]
                if obj_conf_gt == 1:
                    box_x_offset_in_cell = target[i, j, C + 1]
                    box_y_offset_in_cell = target[i, j, C + 2]
                    box_width_rel_img = target[i, j, C + 3]
                    box_height_rel_img = target[i, j, C + 4]

                    center_x_rel_img = (j + box_x_offset_in_cell) / S
                    center_y_rel_img = (i + box_y_offset_in_cell) / S

                    x1 = center_x_rel_img - (box_width_rel_img / 2)
                    y1 = center_y_rel_img - (box_height_rel_img / 2)
                    x2 = center_x_rel_img + (box_width_rel_img / 2)
                    y2 = center_y_rel_img + (box_height_rel_img / 2)

                    class_probs = target[i, j, B * 5: B * 5 + C]
                    class_idx = torch.argmax(class_probs).item()

                    true_boxes.append([
                        x1.item(), y1.item(), x2.item(), y2.item(),
                        class_idx
                    ])
    return true_boxes


def calculate_map(pred_boxes, true_boxes, num_classes, iou_threshold=0.5, class_names=None):
    """
    Calculates mAP (mean Average Precision) for object detection.
    Based on Pascal VOC 2007 evaluation (single IoU threshold).

    Args:
        pred_boxes (list): List of predicted bounding boxes
        true_boxes (list): List of ground truth bounding boxes
        num_classes (int): Number of classes in the dataset
        iou_threshold (float): IoU threshold for considering a match
        class_names (list): List of class names for printing results

    Returns:
        float: mean Average Precision
    """
    ap_per_class = collections.defaultdict(list)
    class_wise_true_positives = collections.defaultdict(int)
    class_wise_false_positives = collections.defaultdict(int)
    class_wise_gt_counts = collections.defaultdict(int)

    for gt_box in true_boxes:
        class_idx = gt_box[4]
        class_wise_gt_counts[class_idx] += 1

    pred_boxes_sorted = sorted(pred_boxes, key=lambda x: x[4], reverse=True)

    for class_id in range(num_classes):  # Changed from C to num_classes
        current_class_preds = [p for p in pred_boxes_sorted if p[6] == class_id]
        current_class_gts = [g for g in true_boxes if g[4] == class_id]

        tp_list = torch.zeros(len(current_class_preds))
        fp_list = torch.zeros(len(current_class_preds))

        matched_gts = torch.zeros(len(current_class_gts))

        for det_idx, detection in enumerate(current_class_preds):
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(current_class_gts):
                if matched_gts[gt_idx] == 0:
                    iou = bbox_iou(
                        torch.tensor(detection[0:4]),
                        torch.tensor(gt[0:4]),
                        x1y1x2y2=True
                    )
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

            if best_iou >= iou_threshold:
                tp_list[det_idx] = 1
                matched_gts[best_gt_idx] = 1
            else:
                fp_list[det_idx] = 1

        tp_cumsum = torch.cumsum(tp_list, dim=0)
        fp_cumsum = torch.cumsum(fp_list, dim=0)

        recalls = tp_cumsum / (class_wise_gt_counts[class_id] + 1e-16)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-16)

        recalls = torch.cat((torch.tensor([0.0]), recalls))
        precisions = torch.cat((torch.tensor([1.0]), precisions))

        ap = torch.trapz(precisions, recalls)

        if class_names:
            print(f"Class: {class_names[class_id]}, AP: {ap.item():.4f}, GTs: {class_wise_gt_counts[class_id]}")
        ap_per_class[class_id] = ap.item()

    if not ap_per_class:
        return 0.0
    mAP = sum(ap_per_class.values()) / len(ap_per_class)

    return mAP