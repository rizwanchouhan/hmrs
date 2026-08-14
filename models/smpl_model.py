"""
Detailed SMPL Model Wrapper using SMPLX library
Author: Your Name
"""

import os
import torch
import torch.nn as nn
from smplx import SMPL
from typing import Optional, Tuple, Dict

class SMPLModel(nn.Module):
    """
    A wrapper around the SMPL body model (from smplx library) that takes shape and pose parameters
    and returns 3D body mesh and joint locations.
    """

    def __init__(self, model_path: str, gender: str = 'neutral', device: str = 'cpu'):
        """
        Initialize the SMPL model.

        Args:
            model_path (str): Path to the folder containing the SMPL model files.
            gender (str): One of ['male', 'female', 'neutral'].
            device (str): 'cpu' or 'cuda' depending on hardware availability.
        """
        super(SMPLModel, self).__init__()

        # Check for valid gender
        assert gender in ['male', 'female', 'neutral'], "Invalid gender selected for SMPL"

        self.model_path = model_path
        self.gender = gender
        self.device = device

        # Initialize SMPL layer
        self.smpl = SMPL(
            model_path=model_path,
            gender=gender,
            batch_size=1  # can be overridden during forward
        ).to(device)

        # Model constants
        self.num_joints = self.smpl.NUM_JOINTS  # Usually 24
        self.num_vertices = self.smpl.get_num_verts()  # Usually 6890

        print(f"[INFO] Loaded SMPL model with {self.num_joints} joints and {self.num_vertices} vertices.")

    def forward(
        self,
        betas: torch.Tensor,
        body_pose: torch.Tensor,
        global_orient: torch.Tensor,
        transl: Optional[torch.Tensor] = None,
        return_full_output: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor] or Dict[str, torch.Tensor]:
        """
        Run forward pass through the SMPL model.

        Args:
            betas (Tensor): Shape parameters [B, 10]
            body_pose (Tensor): Axis-angle pose [B, 69] (23 joints x 3)
            global_orient (Tensor): Root joint orientation [B, 3]
            transl (Tensor, optional): Global translation [B, 3]
            return_full_output (bool): If True, return the full SMPLXOutput object

        Returns:
            vertices (Tensor): [B, 6890, 3] body mesh
            joints (Tensor): [B, 24, 3] joint locations
        """
        output = self.smpl(
            betas=betas,
            body_pose=body_pose,
            global_orient=global_orient,
            transl=transl,
            return_verts=True
        )

        if return_full_output:
            return output
        return output.vertices, output.joints

    def get_specific_joint(self, joints: torch.Tensor, joint_idx: int = 0) -> torch.Tensor:
        """
        Extract a specific joint location.

        Args:
            joints (Tensor): Joint locations [B, 24, 3]
            joint_idx (int): Index of joint to extract

        Returns:
            Tensor: [B, 3] location of the specified joint
        """
        assert joint_idx < self.num_joints, "Joint index out of range."
        return joints[:, joint_idx, :]

    def save_mesh_to_file(self, vertices: torch.Tensor, filename: str):
        """
        Save the 3D mesh to an .obj file (basic).

        Args:
            vertices (Tensor): [B, 6890, 3]
            filename (str): File to save the mesh
        """
        vertices = vertices[0].cpu().numpy()
        with open(filename, 'w') as f:
            for v in vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        print(f"[INFO] Saved mesh to {filename}")

    def print_model_info(self):
        """
        Print summary of model configuration.
        """
        print("====== SMPL Model Info ======")
        print(f"Gender: {self.gender}")
        print(f"Device: {self.device}")
        print(f"Number of joints: {self.num_joints}")
        print(f"Number of vertices: {self.num_vertices}")
        print("=============================")

# Example usage and test
if __name__ == "__main__":
    # Initialize dummy inputs
    batch_size = 2
    betas = torch.randn(batch_size, 10)
    body_pose = torch.randn(batch_size, 69)
    global_orient = torch.randn(batch_size, 3)
    transl = torch.randn(batch_size, 3)

    # Set model path
    model_path = '/path/to/smpl/model'
    gender = 'neutral'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize model
    smpl_model = SMPLModel(model_path=model_path, gender=gender, device=device)
    smpl_model.print_model_info()

    # Run forward pass
    vertices, joints = smpl_model(
        betas=betas.to(device),
        body_pose=body_pose.to(device),
        global_orient=global_orient.to(device),
        transl=transl.to(device)
    )

    print("Vertices shape:", vertices.shape)  # [B, 6890, 3]
    print("Joints shape:", joints.shape)      # [B, 24, 3]

    # Extract and print a specific joint (e.g., pelvis)
    pelvis = smpl_model.get_specific_joint(joints, joint_idx=0)
    print("Pelvis position:", pelvis)

    # Optionally save the mesh
    smpl_model.save_mesh_to_file(vertices, filename="smpl_output.obj")
