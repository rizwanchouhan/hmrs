# Import necessary standard libraries and modules
from cmath import pi  # Mathematical constant pi from complex math library
import imp             # Deprecated, used for importing modules programmatically
import re              # Regular expressions for pattern matching
from typing import Optional  # For type hinting optional parameters
from dataclasses import dataclass  # For data container classes

import os              # Operating system interfaces
import json            # JSON data handling
import pickle          # Serialization and deserialization of Python objects

# Import scientific and deep learning libraries
import numpy as np     # Numerical operations on arrays
import torch           # Core PyTorch library for tensor computations
import torch.nn as nn  # Neural network modules in PyTorch

# Import torchvision for pre-built computer vision models
import torchvision.models as models
from torchvision.models._utils import IntermediateLayerGetter

# Import SMPL-X related body model libraries
import smplx
from smplx import body_models
from smplx import SMPL as _SMPL
from smplx import MANO as _MANO
from smplx import SMPLX as _SMPLX
from smplx import SMPLXLayer, MANOLayer, FLAMELayer
from smplx.lbs import batch_rodrigues, batch_rigid_transform, transform_mat
from smplx.body_models import SMPLXOutput
from smplx.lbs import vertices2joints, blend_shapes

from collections import namedtuple  # For creating tuple-like classes

# Import project-specific modules (assumed to be part of the project)
from main import path_config, constants
from utils import pose_tracker

# ---------------------------------------------
# Define the FeatureExtractor class that uses ResNet-50 as backbone
# and outputs multi-scale feature maps from intermediate layers.
# ---------------------------------------------

class FeatureExtractor(nn.Module):
    """
    FeatureExtractor class leverages a pretrained ResNet-50 network as a backbone model.
    It extracts feature maps from three different stages in the ResNet network, corresponding
    to different spatial resolutions (56x56, 28x28, and 14x14). 
    
    This multi-scale feature extraction is useful for tasks such as object detection, segmentation,
    or pose estimation, where different levels of detail are important.
    
    Attributes:
        backbone (IntermediateLayerGetter): Helper module to extract intermediate layers' outputs.
        reduce_56 (nn.Conv2d): 1x1 convolution to reduce channels of feature map at 56x56 resolution.
        reduce_28 (nn.Conv2d): 1x1 convolution to reduce channels of feature map at 28x28 resolution.
        reduce_14 (nn.Conv2d): 1x1 convolution to reduce channels of feature map at 14x14 resolution.
    """

    def __init__(self, pretrained: bool = True):
        """
        Initialize the FeatureExtractor.
        
        Args:
            pretrained (bool): If True, loads ResNet-50 pretrained on ImageNet dataset.
        """
        # Call the parent class (nn.Module) constructor to initialize internal state
        super(FeatureExtractor, self).__init__()

        # Load a ResNet-50 model, optionally with pretrained weights on ImageNet
        resnet = models.resnet50(pretrained=pretrained)

        # Use IntermediateLayerGetter to access outputs of intermediate layers in ResNet
        # The layers chosen are 'layer1', 'layer2', and 'layer3' which correspond to feature maps
        # at different spatial scales. The keys 'feat_56', 'feat_28', and 'feat_14' are user-defined names.
        self.backbone = IntermediateLayerGetter(
            resnet,
            return_layers={
                'layer1': 'feat_56',  # Layer 1 output (56x56 resolution feature map)
                'layer2': 'feat_28',  # Layer 2 output (28x28 resolution feature map)
                'layer3': 'feat_14'   # Layer 3 output (14x14 resolution feature map)
            }
        )

        # Define 1x1 convolution layers to reduce the channel dimension of each feature map to 256.
        # This helps unify the number of channels across different resolutions for easier processing later.
        self.reduce_56 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)
        self.reduce_28 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.reduce_14 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass to compute the feature maps.
        
        Args:
            x (torch.Tensor): Input image tensor of shape [B, C, H, W], 
                              where B is batch size, C is number of channels (3), 
                              H and W are height and width of the image.
                              
        Returns:
            dict: Dictionary with keys 'feat_56', 'feat_28', 'feat_14' corresponding
                  to feature maps at different spatial resolutions, each with shape
                  [B, 256, H', W'] where H' and W' depend on the scale.
        """
        # Extract feature maps from backbone network
        features = self.backbone(x)

        # Apply 1x1 convolutions to reduce channels for each feature map
        feat_56 = self.reduce_56(features['feat_56'])  # Feature map at 56x56 resolution with 256 channels
        feat_28 = self.reduce_28(features['feat_28'])  # Feature map at 28x28 resolution with 256 channels
        feat_14 = self.reduce_14(features['feat_14'])  # Feature map at 14x14 resolution with 256 channels

        # Return a dictionary of processed feature maps
        return {
            "feat_56": feat_56,
            "feat_28": feat_28,
            "feat_14": feat_14
        }

# ---------------------------------------------
# Example usage of the FeatureExtractor class.
# This is typically placed inside a main guard to allow direct execution.
# ---------------------------------------------

if __name__ == "__main__":
    # Create a dummy input tensor representing a batch of 2 RGB images of size 224x224
    input_tensor = torch.randn(2, 3, 224, 224)  # Shape: [batch_size=2, channels=3, height=224, width=224]

    # Initialize the FeatureExtractor with pretrained weights enabled
    feature_extractor = FeatureExtractor(pretrained=True)

    # Use the feature extractor to get multi-scale features from the input tensor
    features = feature_extractor(input_tensor)

    # Print the shapes of the extracted feature maps to verify dimensions
    print("Extracted feature map shapes:")
    for key, value in features.items():
        print(f"{key}: {value.shape}")

    # Expected output shapes:
    # feat_56: torch.Size([2, 256, 56, 56])
    # feat_28: torch.Size([2, 256, 28, 28])
    # feat_14: torch.Size([2, 256, 14, 14])
