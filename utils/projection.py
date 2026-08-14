import os
import cv2
import torch
import subprocess
import numpy as np
import os.path as osp
from collections import OrderedDict

# Assuming these utilities are defined in separate modules
from utils.smooth_bbox_utils import get_all_bbox_params
from utils.image_utils import get_single_image_crop_demo


# utils/projection.py

import torch


def weak_perspective_projection(points_3d, scale, translation):
    """
    Apply weak perspective projection to 3D points.
    
    Args:
        points_3d (torch.Tensor): [B, N, 3] 3D mesh/joint points
        scale (torch.Tensor): [B, 1] scalar scale s
        translation (torch.Tensor): [B, 2] translation vector [tx, ty]

    Returns:
        points_2d (torch.Tensor): [B, N, 2] projected 2D points
    """
    # Extract the x and y coordinates from 3D points
    xy_coords = points_3d[:, :, :2]
    
    # Reshape scale for broadcasting
    scale_reshaped = scale.unsqueeze(-1)  # [B, 1, 1]

    # Apply scaling to x, y coordinates
    scaled_points = scale_reshaped * xy_coords

    # Reshape translation for broadcasting
    translation_reshaped = translation.unsqueeze(1)  # [B, 1, 2]

    # Add translation to scaled points
    points_2d = scaled_points + translation_reshaped

    return points_2d


def orthographic_projection(points_3d, scale=1.0):
    """
    Orthographic projection for visualization or normalization.
    
    Args:
        points_3d (torch.Tensor): [B, N, 3]
        scale (float, optional): Scaling factor. Defaults to 1.0.

    Returns:
        points_2d (torch.Tensor): [B, N, 2]
    """
    # Extract x, y coordinates
    xy_coords = points_3d[:, :, :2]
    
    # Apply scaling
    points_2d = xy_coords * scale
    
    return points_2d


def normalize_screen_coordinates(points_2d, w, h):
    """
    Normalize 2D screen coordinates to [-1, 1].

    Args:
        points_2d (torch.Tensor): [B, N, 2]
        w (int): Image width
        h (int): Image height

    Returns:
        torch.Tensor: Normalized points in [-1, 1]
    """
    # Clone input tensor to keep original unchanged
    out = points_2d.clone()
    
    # Normalize x coordinates from [0, w] to [-1, 1]
    out[..., 0] = 2.0 * (out[..., 0] / w) - 1.0
    
    # Normalize y coordinates from [0, h] to [-1, 1]
    out[..., 1] = 2.0 * (out[..., 1] / h) - 1.0
    
    return out


def denormalize_screen_coordinates(norm_points, w, h):
    """
    Denormalize 2D points from [-1, 1] back to screen coordinates.

    Args:
        norm_points (torch.Tensor): [B, N, 2] normalized points in [-1, 1]
        w (int): Image width
        h (int): Image height

    Returns:
        torch.Tensor: Denormalized points in pixel coordinates
    """
    out = norm_points.clone()
    
    # Reverse normalization for x
    out[..., 0] = ((out[..., 0] + 1.0) * 0.5) * w
    
    # Reverse normalization for y
    out[..., 1] = ((out[..., 1] + 1.0) * 0.5) * h
    
    return out


def convert_points_to_numpy(points_tensor):
    """
    Convert a torch.Tensor of points to a numpy array.

    Args:
        points_tensor (torch.Tensor): [B, N, 2] or [B, N, 3]

    Returns:
        numpy.ndarray: Converted numpy array
    """
    # Move to CPU if on GPU and convert to numpy
    if points_tensor.is_cuda:
        points_tensor = points_tensor.cpu()
    return points_tensor.detach().numpy()


# Example usage
if __name__ == "__main__":
    # Example 3D points (batch size 2, 2 points each for simplicity)
    points_3d = torch.tensor([
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
    ], dtype=torch.float32)

    # Example scale and translation
    scale = torch.tensor([1.0, 2.0], dtype=torch.float32)  # [B]
    translation = torch.tensor([[10.0, 20.0], [30.0, 40.0]], dtype=torch.float32)  # [B, 2]

    # Apply weak perspective projection
    points_2d_weak = weak_perspective_projection(points_3d, scale, translation)
    print("Weak Perspective Projection:\n", points_2d_weak)

    # Apply orthographic projection with default scale
    points_2d_ortho = orthographic_projection(points_3d)
    print("Orthographic Projection:\n", points_2d_ortho)

    # Normalize screen coordinates for an image of width 640 and height 480
    w, h = 640, 480
    points_2d_normalized = normalize_screen_coordinates(points_2d_weak, w, h)
    print("Normalized Screen Coordinates:\n", points_2d_normalized)

    # Denormalize the normalized points back to pixel coordinates
    points_2d_denormalized = denormalize_screen_coordinates(points_2d_normalized, w, h)
    print("Denormalized Screen Coordinates:\n", points_2d_denormalized)

    # Convert the denormalized points to numpy for further processing
    points_numpy = convert_points_to_numpy(points_2d_denormalized)
    print("Points as numpy array:\n", points_numpy)

    # Additional: Calculate bounding box around projected points
    min_xy = torch.min(points_2d_weak, dim=1).values  # [B, 2]
    max_xy = torch.max(points_2d_weak, dim=1).values  # [B, 2]
    bbox_sizes = max_xy - min_xy  # Width and height for each batch
    print("Bounding box min coordinates:\n", min_xy)
    print("Bounding box max coordinates:\n", max_xy)
    print("Bounding box sizes:\n", bbox_sizes)
