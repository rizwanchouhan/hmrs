import os
import json
import pickle
import re
from cmath import pi
from typing import Optional, List
from dataclasses import dataclass
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# SMPL-X imports
import smplx
from smplx import SMPL as _SMPL
from smplx import MANO as _MANO
from smplx import SMPLX as _SMPLX
from smplx import SMPLXLayer, MANOLayer, FLAMELayer
from smplx.lbs import (
    batch_rodrigues,
    batch_rigid_transform,
    transform_mat,
    vertices2joints,
    blend_shapes
)
from smplx.body_models import SMPLXOutput

# Local project utilities
from main import path_config, constants
from utils import pose_tracker
from utils.projection import project_verts_to_image
from utils.mesh import subdivide_mesh

# Custom model components
from models.graph_conv import SpiralGraphConv
from models.fusion import FeatureFusion


class HierarchicalMeshRefiner(nn.Module):
    """
    Coarse-to-fine hierarchical mesh refinement module.
    Takes initial mesh, refines geometry over multiple stages
    using both mesh and image features.
    """

    def __init__(self, num_stages: int, spiral_indices_list: List[torch.Tensor],
                 feature_dims: List[int], global_dim: int):
        """
        Args:
            num_stages (int): Number of refinement stages.
            spiral_indices_list (List[Tensor]): Spiral conv indices per stage.
            feature_dims (List[int]): Feature dimensions at each stage.
            global_dim (int): Global image feature dimension.
        """
        super(HierarchicalMeshRefiner, self).__init__()

        self.num_stages = num_stages
        self.fusions = nn.ModuleList()
        self.refiners = nn.ModuleList()

        for stage in range(num_stages):
            spiral_idx = spiral_indices_list[stage]
            feat_dim = feature_dims[stage]

            fusion = FeatureFusion(global_dim=global_dim, local_dim=feat_dim)
            gconv = SpiralGraphConv(in_channels=feat_dim, out_channels=feat_dim, spiral_indices=spiral_idx)
            displacement_head = nn.Linear(feat_dim, 3)

            self.fusions.append(fusion)
            self.refiners.append(nn.ModuleDict({
                'gconv': gconv,
                'regressor': displacement_head
            }))

    def forward(self, mesh_verts_init, mesh_feats_init, image_feats_pyramid,
                global_feat, project_func, spiral_maps):
        """
        Refines the mesh vertices through multiple stages.

        Args:
            mesh_verts_init: Tensor [B, V0, 3] Initial mesh vertices.
            mesh_feats_init: Tensor [B, V0, C] Initial vertex features.
            image_feats_pyramid: Dict[str, Tensor] Feature maps at multiple resolutions.
            global_feat: Tensor [B, D] Global pooled image feature.
            project_func: Callable that maps 3D verts -> 2D image coords.
            spiral_maps: List of vertex maps per stage for subdivision.

        Returns:
            Tensor: [B, VN, 3] Final refined mesh vertices.
        """
        verts = mesh_verts_init
        feats = mesh_feats_init

        for stage in range(self.num_stages):
            # 1. Upsample mesh vertices
            verts = subdivide_mesh(verts, spiral_maps[stage])

            # 2. Upsample features
            feats = F.interpolate(feats.transpose(1, 2),
                                  size=verts.shape[1],
                                  mode='nearest').transpose(1, 2)

            # 3. Project vertices to 2D
            uv_coords = project_func(verts)

            # 4. Sample image features at projected coords
            sampled_feats = self.sample_image_features(uv_coords, image_feats_pyramid)

            # 5. Concatenate mesh & image features
            fused_input = torch.cat([feats, sampled_feats], dim=-1)

            # 6. Fuse with global image features
            fused_feats = self.fusions[stage](fused_input, global_feat)

            # 7. Apply graph convolution
            refined_feats = self.refiners[stage]['gconv'](fused_feats)

            # 8. Predict displacements and update mesh
            delta_verts = self.refiners[stage]['regressor'](refined_feats)
            verts = verts + delta_verts
            feats = refined_feats  # Propagate to next stage

        return verts

    def sample_image_features(self, uv_coords: torch.Tensor, feat_pyramid: dict) -> torch.Tensor:
        """
        Bilinearly samples features from multi-scale image pyramids.

        Args:
            uv_coords (Tensor): [B, V, 2] Coordinates in [-1, 1] normalized space.
            feat_pyramid (dict): Multi-resolution feature maps.

        Returns:
            Tensor: [B, V, C] Sampled image features.
        """
        sampled_feats = []

        for k in ['feat_14', 'feat_28', 'feat_56']:
            feat_map = feat_pyramid[k]  # [B, C, H, W]
            B, C, H, W = feat_map.size()

            # Normalize UV coordinates to [-1, 1] for grid_sample
            uv = uv_coords.clone()
            uv = uv.unsqueeze(2)  # [B, V, 1, 2]

            # Grid sample expects coordinates in [-1, 1]
            grid = uv

            # Sample image features using grid_sample
            sampled = F.grid_sample(feat_map, grid, mode='bilinear', align_corners=True)  # [B, C, V, 1]
            sampled = sampled.squeeze(-1).permute(0, 2, 1)  # [B, V, C]
            sampled_feats.append(sampled)

        return torch.cat(sampled_feats, dim=-1)  # [B, V, C_total]


# Example usage block
if __name__ == "__main__":
    # Configuration
    B = 2
    num_vertices = 6890
    num_stages = 3
    feature_dims = [256, 128, 64]
    global_dim = 256
    num_neighbors = 10

    # Dummy data creation
    mesh_verts_init = torch.randn(B, num_vertices, 3)
    mesh_feats_init = torch.randn(B, num_vertices, 256)

    image_feats_pyramid = {
        'feat_14': torch.randn(B, 256, 14, 14),
        'feat_28': torch.randn(B, 256, 28, 28),
        'feat_56': torch.randn(B, 256, 56, 56),
    }

    global_feat = torch.randn(B, global_dim)
    spiral_indices_list = [
        torch.randint(0, num_vertices, (num_vertices, num_neighbors)) for _ in range(num_stages)
    ]
    spiral_maps = [
        torch.randint(0, num_vertices, (num_vertices,)) for _ in range(num_stages)
    ]

    # Dummy projection function: drop z-axis
    def project_func(verts):
        return 2.0 * (verts[:, :, :2] - verts[:, :, :2].min()) / (
            verts[:, :, :2].max() - verts[:, :, :2].min()
        ) - 1.0

    # Initialize and run model
    model = HierarchicalMeshRefiner(
        num_stages=num_stages,
        spiral_indices_list=spiral_indices_list,
        feature_dims=feature_dims,
        global_dim=global_dim
    )

    refined_verts = model(mesh_verts_init, mesh_feats_init, image_feats_pyramid,
                          global_feat, project_func, spiral_maps)

    print("Refined vertices shape:", refined_verts.shape)
