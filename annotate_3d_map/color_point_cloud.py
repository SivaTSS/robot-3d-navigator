# color_point_cloud.py

import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
import json
import gc
from scipy.spatial import cKDTree

def main():
    # Paths (modify these paths if your directory structure is different)
    input_ply = "output_model_points_full.ply"  # Input point cloud
    annotated_ply = "output_model_points_annotated.ply"  # Output annotated point cloud
    annotations_file = "annotations.json"

    # Load annotations from JSON
    if not os.path.exists(annotations_file):
        print(f"Error: Annotations file {annotations_file} does not exist. Please run generate_annotations.py first.")
        return

    with open(annotations_file, 'r') as f:
        combined_annotations = json.load(f)

    # Access 'aggregated_annotations' -> 'objects'
    if ('aggregated_annotations' not in combined_annotations or
        'objects' not in combined_annotations['aggregated_annotations'] or
        not combined_annotations['aggregated_annotations']['objects']):
        print("No objects found in aggregated annotations. Exiting.")
        return

    objects = combined_annotations['aggregated_annotations']['objects']

    # Load the point cloud
    try:
        pcd = o3d.io.read_point_cloud(input_ply)
        if not pcd.has_points():
            print(f"Error: Point cloud file {input_ply} has no points.")
            return
        print(f"Point cloud loaded from {input_ply}")
    except Exception as e:
        print(f"Error loading point cloud: {e}")
        return

    # Ensure the point cloud has colors
    if not pcd.has_colors():
        print("Point cloud has no colors. Initializing to white.")
        num_points = np.asarray(pcd.points).shape[0]
        vertex_colors = np.ones((num_points, 3))
    else:
        vertex_colors = np.asarray(pcd.colors).copy()

    # Reset all point colors to white before coloring annotations
    vertex_colors[:] = [1.0, 1.0, 1.0]

    # Convert point cloud to numpy array
    print("Building KDTree from point cloud vertices...")
    points = np.asarray(pcd.points)

    # Build cKDTree from point cloud vertices
    kdtree = cKDTree(points)

    # Define the radius within which to color points (adjust based on your scale)
    coloring_radius = 0.3  # meters, adjust as needed

    # Define the color map once outside the loop
    color_map = {
        "chair": [1, 0, 0],          # Red
        "couch": [0, 1, 0],          # Green
        "bed": [0, 0, 1],            # Blue
        "dining table": [1, 1, 0],   # Yellow
        "toilet": [1, 0, 1],         # Magenta
        "tv": [0, 1, 1],             # Cyan
        "laptop": [0.5, 0, 0],       # Dark Red
        "mouse": [0, 0.5, 0],         # Dark Green
        "remote": [0, 0, 0.5],        # Dark Blue
        "keyboard": [0.5, 0.5, 0],    # Olive
        "microwave": [0.5, 0, 0.5],   # Purple
        "oven": [0, 0.5, 0.5],        # Teal
        "toaster": [0.75, 0.75, 0],  # Yellow-Orange
        "sink": [0.75, 0, 0.75],      # Pink
        "refrigerator": [0, 0.75, 0.75], # Light Blue
        "book": [0.5, 0.25, 0],       # Brown
        "clock": [0.25, 0.5, 0],      # Light Green
        "vase": [0.25, 0, 0.5],       # Indigo
        "scissors": [0.5, 0, 0],      # Dark Red
        "teddy bear": [0.75, 0.5, 0],# Orange
        "hair dryer": [0.25, 0.25, 0],# Olive
        "toothbrush": [0.25, 0, 0],  # Dark Pink
        "potted plant": [0, 0.25, 0], # Dark Green
        "cup": [0, 0.25, 0.5],        # Light Blue
        "bowl": [0.5, 0, 0.25]        # Maroon
    }

    # Assign colors to the points within the bounding box of each annotation
    print("Assigning colors to point cloud points based on detected objects...")
    for obj in tqdm(objects, desc="Annotating objects"):
        try:
            position = np.array(obj['position'], dtype=np.float64)
            label = obj['class']
            # Define the color based on class
            color = color_map.get(label, [1.0, 1.0, 1.0])  # Default to white if class not in map

            # Query the KDTree for points within the specified radius
            idxs = kdtree.query_ball_point(position, coloring_radius)
            if not idxs:
                print(f"Warning: No points found within radius {coloring_radius} for object '{label}' at position {position}.")
                continue

            # Assign color to these points
            vertex_colors[idxs, :] = color
        except Exception as e:
            print(f"Error processing object {obj}: {e}")

    # Set the updated point colors back to the point cloud
    pcd.colors = o3d.utility.Vector3dVector(vertex_colors)

    # Save the annotated point cloud
    try:
        o3d.io.write_point_cloud(annotated_ply, pcd)
        print(f"Annotated point cloud saved to {annotated_ply}")
    except Exception as e:
        print(f"Error saving annotated point cloud: {e}")

    # Clean up memory
    gc.collect()

if __name__ == "__main__":
    main()
