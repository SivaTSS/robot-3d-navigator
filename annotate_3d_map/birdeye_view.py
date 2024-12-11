import os
import json
import cv2
import numpy as np
import open3d as o3d

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    :param box1: Tuple (x1, y1, x2, y2)
    :param box2: Tuple (x1, y1, x2, y2)
    :return: IoU value
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_width = max(0, x2 - x1 + 1)
    inter_height = max(0, y2 - y1 + 1)
    inter_area = inter_width * inter_height

    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    denominator = float(box1_area + box2_area - inter_area)
    iou = (inter_area / denominator) if denominator > 0 else 0.0
    return iou

def non_max_suppression_bbox(boxes, labels, iou_threshold):
    """
    Apply Non-Max Suppression (NMS) on bounding boxes.
    Since no confidence score is provided, boxes are ranked by area.
    :param boxes: List of bounding boxes [(x1, y1, x2, y2), ...]
    :param labels: List of labels corresponding to each bounding box
    :param iou_threshold: IoU threshold for suppression
    :return: Tuple of (kept_boxes, kept_labels)
    """
    if len(boxes) == 0:
        return [], []

    boxes = np.array(boxes).astype(float)
    labels = np.array(labels)

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # Sort by area (descending), larger boxes first
    idxs = np.argsort(-areas)

    pick = []
    while len(idxs) > 0:
        current = idxs[0]
        pick.append(current)

        if len(idxs) == 1:
            break

        # Compute IoU of the picked box with the rest
        ious = []
        for i in idxs[1:]:
            iou = calculate_iou(boxes[current], boxes[i])
            ious.append(iou)
        ious = np.array(ious)

        # Keep boxes with IoU less than threshold
        remaining_idxs = np.where(ious < iou_threshold)[0]
        idxs = idxs[remaining_idxs + 1]

    kept_boxes = boxes[pick].tolist()
    kept_labels = labels[pick].tolist()
    return kept_boxes, kept_labels

def main():
    print("Generating bird's-eye view...")

    output_dir = "output_birdeye"
    os.makedirs(output_dir, exist_ok=True)

    # Load point cloud map
    map_ply = "output_3d_map/output_model_points_full.ply"
    if not os.path.exists(map_ply):
        print("Error: Map file not found. Run reconstruction first.")
        return

    pcd = o3d.io.read_point_cloud(map_ply)
    if not pcd.has_points():
        print("Error: No points in map.")
        return
    points = np.asarray(pcd.points)

    # Determine XY bounding box of the scene
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])

    # Scale: 100 pixels per meter
    scale = 100
    width = int((max_x - min_x) * scale) + 1
    height = int((max_y - min_y) * scale) + 1
    if width <= 0 or height <= 0:
        print("Invalid map dimensions.")
        return

    # Create visualization image (white background)
    bird_img_vis = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Create binary map image
    # Binary image: Red for obstacles, Blue for floor
    floor_color_bin = (255, 0, 0)      # Blue in BGR
    obstacle_color_bin = (0, 0, 255)   # Red in BGR
    bird_img_bin = np.ones((height, width, 3), dtype=np.uint8) * np.array(floor_color_bin, dtype=np.uint8)

    # Visualization colors
    floor_threshold = 0.1
    floor_color_vis = (255, 200, 200)    # Light pink
    obstacle_color_vis = (200, 200, 255) # Light blue

    # Fill images with floor/obstacle colors based on Z
    for p in points:
        x, y, z = p
        u = int((x - min_x) * scale)
        v = int((max_y - y) * scale)  # invert y
        if 0 <= u < width and 0 <= v < height:
            if z < floor_threshold:
                # Floor
                bird_img_vis[v, u] = floor_color_vis
                bird_img_bin[v, u] = floor_color_bin
            else:
                # Obstacle
                bird_img_vis[v, u] = obstacle_color_vis
                bird_img_bin[v, u] = obstacle_color_bin

    # Load final 3D annotations
    anno_path = "output_3d/final_3d_annotations.json"
    if not os.path.exists(anno_path):
        print("No annotations found. Saving maps without object annotations.")
        cv2.imwrite(os.path.join(output_dir, "birdseye_view_visual.png"), bird_img_vis)
        cv2.imwrite(os.path.join(output_dir, "birdseye_view_binary.png"), bird_img_bin)
        print(f"Birdseye views saved in '{output_dir}' as birdseye_view_visual.png and birdseye_view_binary.png")
        return

    with open(anno_path, 'r') as f:
        annotations = json.load(f)

    # Extract bounding boxes and labels from 3D annotations
    boxes = []
    labels = []
    for obj in annotations['objects']:
        cls = obj['class']
        obj_id = obj['object_id']
        label = f"{cls}{obj_id}"
        bbox_3d = obj.get('bbox_3d', [])

        if len(bbox_3d) != 6:
            print(f"Warning: Object {label} has invalid bbox_3d. Skipping.")
            continue

        min_x_bbox, min_y_bbox, _, max_x_bbox, max_y_bbox, _ = bbox_3d

        # Convert 3D bbox to 2D image coords
        x1 = int((min_x_bbox - min_x) * scale)
        y1 = int((max_y - max_y_bbox) * scale)  # invert y
        x2 = int((max_x_bbox - min_x) * scale)
        y2 = int((max_y - min_y_bbox) * scale)  # invert y

        # Clip to image boundaries
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        if x1 >= x2 or y1 >= y2:
            print(f"Warning: Object {label} has invalid projected bbox. Skipping.")
            continue

        boxes.append((x1, y1, x2, y2))
        labels.append(label)

    # Apply NMS to remove overlapping boxes
    iou_threshold = 0.3
    kept_boxes, kept_labels = non_max_suppression_bbox(boxes, labels, iou_threshold)

    # Draw bounding boxes and labels on visualization image
    for box, label in zip(kept_boxes, kept_labels):
        x1, y1, x2, y2 = map(int, box)  # Ensure coordinates are integers
        cv2.rectangle(bird_img_vis, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

        # Draw label
        font_scale = 0.5
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness_text = 1
        text_size, _ = cv2.getTextSize(label, font, font_scale, thickness_text)
        text_w, text_h = text_size

        # Position text slightly above the box
        text_x, text_y = x1, y1 - 5

        # Adjust if text goes out of boundaries
        if text_x + text_w + 4 > bird_img_vis.shape[1]:
            text_x = x1 - text_w - 4
        if text_y - text_h - 4 < 0:
            text_y = y1 + text_h + 4

        # Background rectangle for text
        cv2.rectangle(bird_img_vis, (text_x, text_y - text_h - 4), (text_x + text_w + 4, text_y), (0, 255, 0), -1)

        # Black text on green background
        cv2.putText(bird_img_vis, label, (text_x + 2, text_y - 2), font, font_scale, (0, 0, 0), thickness_text, cv2.LINE_AA)
    # Remove random noise from obstacles in the binary image
    # Create a mask for obstacles (red pixels)
    obstacle_mask = np.all(bird_img_bin == (0, 0, 255), axis=-1).astype(np.uint8) * 255

    # Morphological opening to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    obstacle_mask_clean = cv2.morphologyEx(obstacle_mask, cv2.MORPH_OPEN, kernel)

    # Reconstruct the binary image with cleaned obstacle mask
    # Wherever obstacle_mask_clean is 255, set red; else set blue
    cleaned_bin_img = np.zeros_like(bird_img_bin)
    cleaned_bin_img[:] = floor_color_bin
    cleaned_bin_img[obstacle_mask_clean == 255] = obstacle_color_bin

    # Save final images
    visual_path = os.path.join(output_dir, "birdseye_view_visual.png")
    binary_path = os.path.join(output_dir, "birdseye_view_binary.png")

    cv2.imwrite(visual_path, bird_img_vis)
    cv2.imwrite(binary_path, cleaned_bin_img)

    print(f"Birdseye views saved in '{output_dir}' as 'birdseye_view_visual.png' and 'birdseye_view_binary.png'.")

if __name__ == "__main__":
    main()
