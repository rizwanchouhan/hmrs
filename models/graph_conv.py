from cmath import pi
import imp
import re
from typing import Optional
from dataclasses import dataclass

import os
import json
import pickle
import numpy as np
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import smplx
from smplx import SMPL as _SMPL
from smplx import MANO as _MANO
from smplx import SMPLX as _SMPLX
from smplx import SMPLXLayer, MANOLayer, FLAMELayer
from smplx import body_models
from smplx.lbs import (
    batch_rodrigues, batch_rigid_transform, transform_mat,
    vertices2joints, blend_shapes
)
from smplx.body_models import SMPLXOutput

from main import path_config, constants
from utils import pose_tracker


class SpiralGraphConv(nn.Module):
    """
    Spiral Graph Convolution block for mesh vertex feature refinement.
    Each vertex gathers features from its spiral neighbors and applies shared MLP.
    """
    def __init__(self, in_channels: int, out_channels: int, spiral_indices: torch.Tensor, dropout: float = 0.0):
        """
        Initializes the SpiralGraphConv module.

        Args:
            in_channels (int): Number of input feature channels per vertex
            out_channels (int): Number of output feature channels per vertex
            spiral_indices (torch.Tensor): Tensor of shape [V, K] with spiral neighbor indices
            dropout (float): Dropout probability
        """
        super(SpiralGraphConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spiral_indices = spiral_indices  # LongTensor [V, K]
        self.K = spiral_indices.shape[1]      # Number of neighbors

        # Linear transformation layer: processes concatenated neighbor features
        self.linear = nn.Linear(in_channels * self.K, out_channels)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for SpiralGraphConv.

        Args:
            x (torch.Tensor): Input vertex features of shape [B, V, C_in]

        Returns:
            torch.Tensor: Refined vertex features of shape [B, V, C_out]
        """
        B, V, C = x.shape
        device = x.device

        # Move spiral indices to the same device as input
        spiral_idx = self.spiral_indices.to(device)  # [V, K]
        spiral_idx = spiral_idx.unsqueeze(0).expand(B, -1, -1)  # [B, V, K]

        # Gather K-neighbor features for each vertex
        x_gathered = torch.gather(x, 1, spiral_idx.unsqueeze(-1).expand(-1, -1, -1, C))  # [B, V, K, C]

        # Reshape gathered features to [B, V, K*C] to feed into the linear layer
        x_gathered = x_gathered.reshape(B, V, -1)

        # Apply linear transformation, dropout, and activation
        out = self.linear(x_gathered)
        out = self.dropout(out)
        out = self.activation(out)

        return out


# Example usage of SpiralGraphConv
if __name__ == "__main__":
    # Configuration parameters
    batch_size = 2
    num_vertices = 6890  # Number of mesh vertices (SMPL-X model)
    num_neighbors = 10   # Number of spiral neighbors
    in_channels = 128    # Feature dimension per vertex (input)
    out_channels = 256   # Feature dimension per vertex (output)
    dropout_rate = 0.2

    # Generate random input features for each vertex in each mesh in the batch
    x = torch.randn(batch_size, num_vertices, in_channels)  # [B, V, C_in]

    # Generate random spiral indices (normally this comes from mesh topology preprocessing)
    spiral_indices = torch.randint(0, num_vertices, (num_vertices, num_neighbors), dtype=torch.long)

    # Initialize the graph convolution module
    graph_conv = SpiralGraphConv(in_channels, out_channels, spiral_indices, dropout=dropout_rate)

    # Apply the graph convolution to input features
    refined_features = graph_conv(x)  # [B, V, C_out]

    # Print output shape for verification
    print("Input features shape:", x.shape)
    print("Refined features shape:", refined_features.shape)

    # Optionally print a few vertex features before and after for inspection
    print("\nSample input features (vertex 0):", x[0, 0, :5])
    print("Sample refined features (vertex 0):", refined_features[0, 0, :5])

