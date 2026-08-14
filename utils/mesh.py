import os
import cv2
import torch
import subprocess
import numpy as np
import os.path as osp
from collections import OrderedDict

from utils.smooth_bbox_utils import get_all_bbox_params
from utils.image_utils import get_single_image_crop_demo


# =========================================
#         Mesh Utility Functions
# =========================================

def subdivide_mesh(verts, upsample_map):
    """
    Subdivide a mesh by upsampling vertices using a precomputed map.

    Args:
        verts (torch.Tensor): Tensor of shape (B, V_low, 3) containing the lower-resolution mesh vertices.
        upsample_map (np.ndarray or torch.Tensor): Index map of shape (V_high,) or (V_high, K) mapping high-res vertices
                                                   to combinations of low-res ones. For this simple case, we assume it's (V_high,).

    Returns:
        torch.Tensor: Upsampled mesh vertices of shape (B, V_high, 3).
    """
    if not isinstance(verts, torch.Tensor):
        raise TypeError("verts must be a torch.Tensor")

    if isinstance(upsample_map, np.ndarray):
        upsample_map = torch.LongTensor(upsample_map).to(verts.device)
    elif not isinstance(upsample_map, torch.Tensor):
        raise TypeError("upsample_map must be a numpy array or torch tensor")

    if upsample_map.dim() != 1:
        raise ValueError("Only 1D upsample maps are supported in this simple version")

    # Apply nearest-neighbor upsampling
    upsampled_verts = verts[:, upsample_map]
    return upsampled_verts


def load_spiral_indices(path):
    """
    Load precomputed spiral indices from a .npz file.

    Args:
        path (str): Path to the .npz file.

    Returns:
        List[torch.LongTensor]: Spiral indices for each level, loaded as PyTorch tensors.
    """
    if not osp.exists(path):
        raise FileNotFoundError(f"Spiral indices file not found at: {path}")

    data = np.load(path, allow_pickle=True)
    spiral_indices = []

    print(f"Loaded spiral index file with {len(data.files)} arrays.")

    for i in range(len(data.files)):
        array = data[f'arr_{i}']
        tensor = torch.LongTensor(array)
        spiral_indices.append(tensor)
        print(f"Loaded spiral index arr_{i}: shape = {tensor.shape}")

    return spiral_indices


def downsample_mesh(verts, downsample_idx):
    """
    Downsample a mesh using a set of index positions.

    Args:
        verts (torch.Tensor): Mesh vertices of shape (B, V_full, 3).
        downsample_idx (np.ndarray or torch.Tensor): Indices of shape (V_small,) to select vertices.

    Returns:
        torch.Tensor: Downsampled vertices of shape (B, V_small, 3).
    """
    if isinstance(downsample_idx, np.ndarray):
        downsample_idx = torch.LongTensor(downsample_idx).to(verts.device)

    if downsample_idx.dim() != 1:
        raise ValueError("downsample_idx must be a 1D array or tensor")

    downsampled_verts = verts[:, downsample_idx]
    return downsampled_verts


def create_mock_spiral_indices(save_path="mock_spiral_indices.npz", num_levels=3, level_size=10):
    """
    Create and save mock spiral index arrays for testing purposes.

    Args:
        save_path (str): Where to save the .npz file.
        num_levels (int): Number of levels (arrays) to create.
        level_size (int): Size of each spiral array.
    """
    spiral_dict = {}
    for i in range(num_levels):
        spiral_dict[f'arr_{i}'] = np.random.randint(0, 100, size=(level_size,))
    np.savez(save_path, **spiral_dict)
    print(f"Mock spiral indices saved at: {save_path}")


# =========================================
#               Main Execution
# =========================================

if __name__ == "__main__":
    # Setup dummy data
    batch_size = 2
    num_low_res_vertices = 4
    num_high_res_vertices = 6
    downsample_count = 2

    # Dummy low-res mesh: (B, V, 3)
    verts = torch.tensor([
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        [[13.0, 14.0, 15.0], [16.0, 17.0, 18.0], [19.0, 20.0, 21.0], [22.0, 23.0, 24.0]]
    ], dtype=torch.float32)

    print(f"Original Vertices Shape: {verts.shape}")

    # Example upsample map: replicate some low-res vertices
    upsample_map = np.array([0, 1, 2, 3, 0, 1])

    # Downsample to first and third vertex
    downsample_idx = np.array([0, 2])

    # Subdivide (upsample)
    print("\n--- Upsampling ---")
    upsampled_verts = subdivide_mesh(verts, upsample_map)
    print("Upsampled Vertices:\n", upsampled_verts)

    # Create and load mock spiral indices
    print("\n--- Spiral Index Loading ---")
    mock_path = "mock_spiral_indices.npz"
    create_mock_spiral_indices(mock_path, num_levels=3, level_size=5)
    spiral_indices = load_spiral_indices(mock_path)

    # Downsample
    print("\n--- Downsampling ---")
    downsampled_verts = downsample_mesh(verts, downsample_idx)
    print("Downsampled Vertices:\n", downsampled_verts)

    print("\nDone.")
