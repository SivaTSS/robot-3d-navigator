import os
import numpy as np
import open3d as o3d
import cv2
from tqdm import tqdm
import gc  # For garbage collection

def load_matrix(file_path):
    """
    Load a 4x4 matrix from a text file.
    """
    matrix = []
    with open(file_path, 'r') as f:
        for line in f:
            # Split by whitespace and convert to float
            row = [float(num) for num in line.strip().split()]
            matrix.append(row)
    return np.array(matrix)

def backproject_depth(depth, intrinsic, downsample_factor=4):
    """
    Backproject depth image to 3D point cloud in camera coordinates.

    Parameters:
        depth (numpy.ndarray): Depth image in meters.
        intrinsic (numpy.ndarray): 4x4 intrinsic matrix.
        downsample_factor (int): Factor by which to downsample the depth image.

    Returns:
        points (numpy.ndarray): Nx3 array of 3D points in camera coordinates.
        uu (numpy.ndarray): Nx1 array of pixel x-coordinates.
        vv (numpy.ndarray): Nx1 array of pixel y-coordinates.
    """
    # Downsample depth and intrinsic matrix
    if downsample_factor > 1:
        depth = depth[::downsample_factor, ::downsample_factor]
        intrinsic = intrinsic.copy()
        intrinsic[0, 0] /= downsample_factor  # Scale fx
        intrinsic[1, 1] /= downsample_factor  # Scale fy
        intrinsic[0, 2] /= downsample_factor  # Scale cx
        intrinsic[1, 2] /= downsample_factor  # Scale cy

    height, width = depth.shape
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    # Create a grid of (u,v) coordinates
    u = np.arange(width)
    v = np.arange(height)
    uu, vv = np.meshgrid(u, v)

    # Flatten the arrays
    uu = uu.flatten()
    vv = vv.flatten()
    depth = depth.flatten()

    # Filter out zero depth
    valid = depth > 0
    uu = uu[valid]
    vv = vv[valid]
    depth = depth[valid]

    # Backproject to camera coordinates
    x = (uu - cx) * depth / fx
    y = (vv - cy) * depth / fy
    z = depth
    points = np.vstack((x, y, z)).T
    return points, uu, vv

def main():
    # Paths (modify these paths if your directory structure is different)
    data_folder_path = "data/scannet_data/scans/scene0000_00/rgbd"
    color_dir = os.path.join(data_folder_path, "color")
    depth_dir = os.path.join(data_folder_path, "depth")
    intrinsic_dir = os.path.join(data_folder_path, "intrinsic")
    pose_dir = os.path.join(data_folder_path, "pose")

    output_ply = "output_model_points.ply"
    output_mesh_ply = "output_model_mesh.ply"

    # Load intrinsic and extrinsic matrices
    intrinsic_depth_path = os.path.join(intrinsic_dir, "intrinsic_depth.txt")
    extrinsic_depth_path = os.path.join(intrinsic_dir, "extrinsic_depth.txt")
    intrinsic_color_path = os.path.join(intrinsic_dir, "intrinsic_color.txt")
    extrinsic_color_path = os.path.join(intrinsic_dir, "extrinsic_color.txt")

    intrinsic_depth = load_matrix(intrinsic_depth_path)
    extrinsic_depth = load_matrix(extrinsic_depth_path)
    intrinsic_color = load_matrix(intrinsic_color_path)
    extrinsic_color = load_matrix(extrinsic_color_path)

    # If extrinsic matrices are not provided, assume identity
    if extrinsic_depth.size == 0:
        extrinsic_depth = np.eye(4)
    if extrinsic_color.size == 0:
        extrinsic_color = np.eye(4)

    # Specify the number of images to process
    # num_images_to_process = 100  # Change this value to control the number of images to process

    # Get list of image indices (assuming filenames are sequential numbers)
    image_files = [f for f in os.listdir(depth_dir) if f.endswith('.png')]
    image_indices = sorted([int(os.path.splitext(f)[0]) for f in image_files])

    # Limit the number of images to process
    # image_indices = image_indices[:num_images_to_process]

    # Initialize an empty point cloud
    pcd = o3d.geometry.PointCloud()

    print(f"Starting 3D reconstruction on {len(image_indices)} images...")

    for idx in tqdm(image_indices, desc="Processing images"):
        try:
            # Paths for current image
            depth_path = os.path.join(depth_dir, f"{idx}.png")
            color_path = os.path.join(color_dir, f"{idx}.jpg")
            pose_path = os.path.join(pose_dir, f"{idx}.txt")

            # Load depth image
            depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_image is None:
                print(f"Warning: Unable to load depth image {depth_path}. Skipping.")
                continue
            # Convert depth to meters (assuming the depth is in millimeters)
            depth = depth_image.astype(np.float32) / 1000.0  # Adjust scaling if necessary

            # Load color image
            color_image = cv2.imread(color_path, cv2.IMREAD_COLOR)
            if color_image is None:
                print(f"Warning: Unable to load color image {color_path}. Skipping.")
                continue
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            # Load pose
            pose = load_matrix(pose_path)

            # Backproject depth to camera coordinates (with downsampling)
            points_camera, uu, vv = backproject_depth(depth, intrinsic_depth, downsample_factor=4)  # More aggressive downsample factor: 8

            # Apply extrinsic_depth to get points in depth camera frame (if necessary)
            points_camera_hom = np.hstack((points_camera, np.ones((points_camera.shape[0], 1))))
            points_depth = (extrinsic_depth @ points_camera_hom.T).T[:, :3]

            # Transform points to world coordinates using the pose
            points_world_hom = np.hstack((points_depth, np.ones((points_depth.shape[0], 1))))
            points_world = (pose @ points_world_hom.T).T[:, :3]

            # Get corresponding colors
            # Ensure that vv and uu are integers and within the image bounds
            h_color, w_color, _ = color_image.shape
            valid_indices = (uu >= 0) & (uu < w_color) & (vv >= 0) & (vv < h_color)
            # Filter points_world and color indices accordingly
            points_world = points_world[valid_indices]
            uu_valid = uu[valid_indices].astype(int)
            vv_valid = vv[valid_indices].astype(int)
            colors = color_image[vv_valid, uu_valid, :] / 255.0  # Normalize to [0,1]

            # Create a temporary point cloud
            temp_pcd = o3d.geometry.PointCloud()
            temp_pcd.points = o3d.utility.Vector3dVector(points_world)
            temp_pcd.colors = o3d.utility.Vector3dVector(colors)

            # Merge with global point cloud
            pcd += temp_pcd

            # Release memory
            del temp_pcd, points_world, points_camera, uu, vv, valid_indices, colors
            gc.collect()

            # Periodically downsample to manage memory
            if len(pcd.points) > 500000:  # Threshold, adjust as necessary
                pcd = pcd.voxel_down_sample(voxel_size=0.02)
                gc.collect()

        except Exception as e:
            print(f"Error processing image {idx}: {e}")
            continue

    if not pcd.points:
        print("No valid points were processed. Exiting.")
        return

    # Final downsampling
    voxel_size = 0.05  # Adjust voxel size to coarsen the 3D model
    print("Final downsampling of the point cloud...")
    try:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    except Exception as e:
        print(f"Error during final downsampling: {e}")
        return

    print(f"Total points after final downsampling: {len(pcd.points)}")

    # Save the point cloud for reference
    try:
        o3d.io.write_point_cloud(output_ply, pcd)
        print(f"Point cloud saved to {output_ply}")
    except Exception as e:
        print(f"Error saving point cloud PLY file: {e}")

    # Estimate normals for the point cloud
    print("Estimating normals for the point cloud...")
    try:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    except Exception as e:
        print(f"Error estimating normals: {e}")
        return

    # Generate a mesh from the point cloud using Poisson Surface Reconstruction
    print("Performing Poisson Surface Reconstruction...")
    try:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        print("Poisson surface reconstruction completed.")
    except Exception as e:
        print(f"Error during Poisson Surface Reconstruction: {e}")
        return

    # Optionally, crop the mesh to focus on areas with high density
    print("Cropping the mesh to remove low-density areas...")
    try:
        densities = np.asarray(densities)
        density_threshold = np.percentile(densities, 5)  # Adjust percentile as needed
        vertices_to_remove = densities < density_threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)
        print(f"Mesh cropped. Remaining vertices: {len(mesh.vertices)}")
    except Exception as e:
        print(f"Error during mesh cropping: {e}")
        return

    # Simplify the mesh to reduce the number of triangles
    print("Simplifying the mesh...")
    try:
        target_number_of_triangles = 300000  # Adjust as needed
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_number_of_triangles)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        print(f"Mesh simplified to {len(mesh.triangles)} triangles.")
    except Exception as e:
        print(f"Error during mesh simplification: {e}")
        return

    # Save the mesh to a file
    try:
        o3d.io.write_triangle_mesh(output_mesh_ply, mesh)
        print(f"Mesh saved to {output_mesh_ply}")
    except Exception as e:
        print(f"Error saving mesh PLY file: {e}")

if __name__ == "__main__":
    main()
