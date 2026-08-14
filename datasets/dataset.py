import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

class HumanMeshDataset(Dataset):
    """
    HumanMeshDataset is a PyTorch Dataset class tailored for human mesh recovery tasks.
    It supports COCO-style 2D keypoints and optional 3D data like SMPL pose and shape,
    3D joints, and even segmentation masks for tasks such as dense pose estimation.

    Each sample contains:
        - Image tensor
        - 2D keypoints [J, 3]
        - (Optional) SMPL pose, shape, and joints_3d
        - (Optional) segmentation mask
    """

    def __init__(self, 
                 data_dir: str, 
                 split: str = 'train', 
                 use_3d: bool = False, 
                 use_masks: bool = False,
                 img_size: int = 224, 
                 augment: bool = True):
        """
        Initializes the dataset loader.

        Args:
            data_dir (str): Root directory of dataset.
            split (str): Dataset split - 'train', 'val', or 'test'.
            use_3d (bool): Whether to include SMPL and 3D joint info.
            use_masks (bool): Whether to include segmentation masks.
            img_size (int): Target image resolution (square).
            augment (bool): Whether to apply image augmentation.
        """
        self.data_dir = data_dir
        self.split = split
        self.use_3d = use_3d
        self.use_masks = use_masks
        self.img_size = img_size
        self.augment = augment

        # Construct paths
        self.img_folder = os.path.join(data_dir, 'images', split)
        self.mask_folder = os.path.join(data_dir, 'masks', split) if use_masks else None
        self.anno_path = os.path.join(data_dir, f'{split}_annot.json')

        # Validate paths
        assert os.path.exists(self.anno_path), f"Annotation file not found at: {self.anno_path}"
        assert os.path.isdir(self.img_folder), f"Image folder not found at: {self.img_folder}"
        if self.use_masks:
            assert os.path.isdir(self.mask_folder), f"Mask folder not found at: {self.mask_folder}"

        # Load annotations into memory
        with open(self.anno_path, 'r') as f:
            self.annotations = json.load(f)

        # Setup transformations
        self.transform = self.build_transform()

    def __len__(self):
        """
        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.annotations)

    def __getitem__(self, idx: int):
        """
        Fetches a data sample by index.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: Dictionary with image, keypoints, and optional 3D and mask data.
        """
        ann = self.annotations[idx]
        img_path = os.path.join(self.img_folder, ann['img_name'])
        image = Image.open(img_path).convert('RGB')

        keypoints_2d = np.array(ann['keypoints_2d']).reshape(-1, 3)
        bbox = ann.get('bbox', self.get_tight_bbox(keypoints_2d))

        # Crop and resize
        image, keypoints_2d = self.crop_and_resize(image, keypoints_2d, bbox)
        image = self.transform(image)

        # Normalize or pad 2D keypoints to [17, 3] for COCO-style compatibility
        keypoints_2d = self.pad_keypoints(keypoints_2d, target_len=17)

        # Create data dictionary
        data = {
            'image': image,
            'keypoints_2d': torch.from_numpy(keypoints_2d).float()
        }

        # Include 3D data if available
        if self.use_3d:
            data['pose'] = torch.tensor(ann['pose']).float()
            data['shape'] = torch.tensor(ann['shape']).float()
            data['joints_3d'] = torch.tensor(ann['joints_3d']).float()

        # Include mask if available
        if self.use_masks and 'mask_name' in ann:
            mask_path = os.path.join(self.mask_folder, ann['mask_name'])
            mask = Image.open(mask_path).convert('L')  # single channel
            mask = mask.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
            mask = mask.resize((self.img_size, self.img_size))
            mask = T.ToTensor()(mask)
            data['mask'] = mask

        return data

    def pad_keypoints(self, keypoints, target_len=17):
        """
        Pads or trims keypoints to a fixed number of joints.

        Args:
            keypoints (np.ndarray): Original keypoints [J, 3].
            target_len (int): Desired number of keypoints.

        Returns:
            np.ndarray: Padded/truncated keypoints.
        """
        num_joints = keypoints.shape[0]
        padded = np.zeros((target_len, 3))
        padded[:min(num_joints, target_len)] = keypoints[:target_len]
        return padded

    def build_transform(self):
        """
        Build the image transformation pipeline.

        Returns:
            torchvision.transforms.Compose: Composed transformations.
        """
        transforms = [T.Resize((self.img_size, self.img_size))]
        if self.augment:
            transforms += [
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                T.RandomRotation(degrees=20)
            ]
        transforms += [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ]
        return T.Compose(transforms)

    def crop_and_resize(self, image, keypoints, bbox):
        """
        Crops the image around a bounding box and resizes to (img_size, img_size).

        Args:
            image (PIL.Image): Input image.
            keypoints (np.ndarray): [J, 3] keypoints.
            bbox (list): [x, y, w, h] format.

        Returns:
            Tuple[PIL.Image, np.ndarray]: Cropped image and adjusted keypoints.
        """
        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2
        scale = max(w, h) * 1.25  # enlarge bbox
        left = cx - scale / 2
        top = cy - scale / 2

        # Crop and resize
        image = image.crop((left, top, left + scale, top + scale))
        image = image.resize((self.img_size, self.img_size))

        # Adjust keypoints
        keypoints[:, 0] = (keypoints[:, 0] - left) * self.img_size / scale
        keypoints[:, 1] = (keypoints[:, 1] - top) * self.img_size / scale

        return image, keypoints

    def get_tight_bbox(self, keypoints):
        """
        Computes a bounding box tightly enclosing the visible keypoints.

        Args:
            keypoints (np.ndarray): [J, 3] keypoints.

        Returns:
            list: [x, y, w, h] bounding box.
        """
        visible = keypoints[:, 2] > 0
        if visible.sum() == 0:
            return [0, 0, self.img_size, self.img_size]

        x_min = keypoints[visible][:, 0].min()
        y_min = keypoints[visible][:, 1].min()
        x_max = keypoints[visible][:, 0].max()
        y_max = keypoints[visible][:, 1].max()
        return [x_min, y_min, x_max - x_min, y_max - y_min]


# Example usage and data loading
if __name__ == "__main__":
    dataset = HumanMeshDataset(
        data_dir='path/to/dataset',
        split='train',
        use_3d=True,
        use_masks=True,
        img_size=224,
        augment=True
    )

    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    for batch in data_loader:
        images = batch['image']          # Tensor [B, 3, H, W]
        keypoints_2d = batch['keypoints_2d']  # Tensor [B, 17, 3]
        print(f"Batch image shape: {images.shape}")
        print(f"2D keypoints shape: {keypoints_2d.shape}")

        if 'pose' in batch:
            print(f"Pose shape: {batch['pose'].shape}")
            print(f"Shape (betas): {batch['shape'].shape}")
            print(f"3D joints shape: {batch['joints_3d'].shape}")

        elif 'mask' in batch:
            print(f"Mask shape: {batch['mask'].shape}")