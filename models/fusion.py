# ============================
# Imports and Dependencies
# ============================

from cmath import pi  # Complex math module; pi is used in angular calculations
import imp  # Legacy module, generally used for dynamic loading (deprecated in favor of importlib)
import re  # Regular expressions for pattern matching
from typing import Optional  # For optional type hinting
from dataclasses import dataclass  # For cleaner, structured data representations

import os  # File path operations
import json  # JSON parsing and exporting
import pickle  # For saving/loading Python objects

# Torch and related modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# SMPLX body models and LBS (linear blend skinning) tools
import smplx
from smplx import body_models
from smplx import SMPL as _SMPL
from smplx import MANO as _MANO
from smplx import SMPLX as _SMPLX
from smplx import SMPLXLayer, MANOLayer, FLAMELayer
from smplx.lbs import batch_rodrigues, batch_rigid_transform, transform_mat
from smplx.lbs import vertices2joints, blend_shapes
from smplx.body_models import SMPLXOutput

from collections import namedtuple  # For tuple-like data containers

# Local project modules
from main import path_config, constants
from utils import pose_tracker


# ============================
# Feature Fusion Module
# ============================

class FeatureFusion(nn.Module):
    """
    FeatureFusion implements a learnable attention-based fusion between global
    image features and local per-vertex features. This is useful when combining
    a holistic representation of the image with fine-grained information per mesh vertex.

    Implements Equations (4)-(6) as described in the original paper.

    Attributes:
        global_dim (int): Dimensionality of global features f_g
        local_dim (int): Dimensionality of local (per-vertex) features f_l
    """

    def __init__(self, global_dim: int, local_dim: int):
        """
        Initializes the FeatureFusion module with learnable projection and attention layers.

        Args:
            global_dim (int): Size of the global feature vector
            local_dim (int): Size of the per-vertex local feature vectors
        """
        super(FeatureFusion, self).__init__()

        # Linear projection to align global feature dimension with local dimension
        self.project_global = nn.Linear(global_dim, local_dim)

        # Attention mechanism: learns how to weight global vs local features
        self.attention_fc = nn.Linear(2 * local_dim, local_dim)

        # Non-linearity to produce weights in [0,1] range
        self.sigmoid = nn.Sigmoid()

    def forward(self, local_feats: torch.Tensor, global_feats: torch.Tensor) -> torch.Tensor:
        """
        Fuses the local and global features using an attention-weighted mechanism.

        Args:
            local_feats (Tensor): Shape [B, V, D_l], local features per mesh vertex
            global_feats (Tensor): Shape [B, D_g], global feature vector per image

        Returns:
            fused_feats (Tensor): Shape [B, V, D_l], fused output features per vertex
        """
        B, V, D_l = local_feats.shape  # Batch size, vertices, local feature dim
        D_g = global_feats.shape[1]    # Global feature dim

        # Step 1: Project global feature to match local dimension
        global_proj = self.project_global(global_feats)  # Shape: [B, D_l]

        # Step 2: Expand global projection to match the number of vertices
        global_proj = global_proj.unsqueeze(1).expand(-1, V, -1)  # Shape: [B, V, D_l]

        # Step 3: Concatenate local and global features at each vertex
        concat = torch.cat([local_feats, global_proj], dim=-1)  # Shape: [B, V, 2*D_l]

        # Step 4: Compute attention weights alpha for fusion
        alpha = self.sigmoid(self.attention_fc(concat))  # Shape: [B, V, D_l]

        # Step 5: Perform weighted fusion using learned attention
        fused = alpha * global_proj + (1 - alpha) * local_feats  # Shape: [B, V, D_l]

        return fused


# ============================
# Example / Testing Block
# ============================

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # ----------------------------
    # Configuration Parameters
    # ----------------------------
    batch_size = 2
    global_dim = 256
    local_dim = 128
    num_vertices = 6890  # Number of SMPLX mesh vertices

    # ----------------------------
    # Generate Dummy Data
    # ----------------------------

    # Simulated per-vertex features (e.g., from GCN or CNN)
    local_feats = torch.randn(batch_size, num_vertices, local_dim)

    # Simulated global features (e.g., from CNN or image encoder)
    global_feats = torch.randn(batch_size, global_dim)

    # ----------------------------
    # Initialize Fusion Module
    # ----------------------------

    fusion_module = FeatureFusion(global_dim=global_dim, local_dim=local_dim)

    # Print architecture
    print(fusion_module)

    # ----------------------------
    # Perform Forward Pass
    # ----------------------------

    fused_feats = fusion_module(local_feats, global_feats)

    # ----------------------------
    # Output Results
    # ----------------------------

    print("Input Local Feature Shape :", local_feats.shape)
    print("Input Global Feature Shape:", global_feats.shape)
    print("Fused Feature Shape       :", fused_feats.shape)

    # Optional: Check min/max values for sanity
    print("Fused Feature Value Range :", fused_feats.min().item(), "to", fused_feats.max().item())
