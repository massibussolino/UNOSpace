import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd

class PanoptesDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, template_offset=5):
        self.dataset_path = dataset_path
        self.imgs_path = os.path.join(dataset_path, 'images')
        self.depth_path = os.path.join(dataset_path, 'depths')
        self.mask_path = os.path.join(dataset_path, 'masks') # Added mask path
        self.labels_path = pd.read_json(os.path.join(dataset_path, 'labels.json'))
        
        self.imgs = sorted(os.listdir(self.imgs_path))
        self.depths = sorted(os.listdir(self.depth_path))
        self.masks = sorted(os.listdir(self.mask_path)) # Added masks list

        self.img_size = 224
        self.num_points = 1024
        self.template_offset = template_offset
        
        # Calculated K matrix from your 1024x1024, 40-deg FOV specs
        self.K = np.array([[1406.71, 0.0, 512.0],
                           [0.0, 1406.71, 512.0],
                           [0.0, 0.0,     1.0]])

        self.rgb_transform = transforms.Compose([
            transforms.ToTensor(), 
        ])

    def __len__(self):
        return len(self.imgs)
        
    def _get_resize_rgb_choose(self, choose, bbox, img_size):
        # (Unchanged from previous version)
        y1, y2, x1, x2 = bbox
        crop_h = y2 - y1
        crop_w = x2 - x1
        
        y_crop = choose // crop_w
        x_crop = choose % crop_w
        
        y_res = (y_crop * (img_size / crop_h)).astype(np.int64)
        x_res = (x_crop * (img_size / crop_w)).astype(np.int64)
        
        y_res = np.clip(y_res, 0, img_size - 1)
        x_res = np.clip(x_res, 0, img_size - 1)
        
        rgb_choose = y_res * img_size + x_res
        return rgb_choose

    def _get_bbox_from_mask(self, mask):
        """Calculates a tight bounding box around the non-zero elements of a mask."""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        # Find the first and last True values
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        
        # Add +1 to the max bounds so slicing works correctly (e.g., mask[y1:y2])
        return y1, y2 + 1, x1, x2 + 1

    def _load_data(self, idx):
        # --- 1. Load Raw Data (RGB, Depth, Mask) ---
        img_path = os.path.join(self.imgs_path, self.imgs[idx])
        rgb_full = Image.open(img_path).convert('RGB')
        
        depth_path = os.path.join(self.depth_path, self.depths[idx])
        # Depth image is expected to store values in meters (e.g. float32 TIFF/EXR).
        # The +15 accounts for the 15 m near-clip-plane offset so that
        # absolute depth = stored_value + 15  [meters].
        # If your files are 16-bit PNG (raw integers), divide by the
        # appropriate scale factor before adding the offset so the result
        # is in metres (matching the BOP pipeline used during training).
        depth_full = np.array(Image.open(depth_path)).astype(np.float32)/65535 * (40-15) + 15  # meters

        # # compute the mean of the px > 0 and < 65535
        # depth_raw = np.array(Image.open(depth_path)).astype(np.float32)
        # valid_depths = depth_raw[(depth_raw > 0) & (depth_raw < 65535)]
        # if len(valid_depths) > 0:
        #     depth_mean = (valid_depths/65535 * (40-15) + 15).mean()
        # else:
        #     depth_mean = 0.0  # Fallback if no valid depths are found
        
        # # Change all the values of depth full with the mean
        # depth_full = np.full_like(depth_full, depth_mean)

        
        mask_path = os.path.join(self.mask_path, self.masks[idx])
        mask_full = np.array(Image.open(mask_path)) # Assuming grayscale/binary mask

        # --- 2. Define Bounding Box Dynamically ---
        # Get the bbox from the mask directly
        y1, y2, x1, x2 = self._get_bbox_from_mask(mask_full)
        bbox = [y1, y2, x1, x2]

        # --- 3. Process Depth & Mask Crops ---
        depth_crop = depth_full[y1:y2, x1:x2]
        mask_crop = mask_full[y1:y2, x1:x2]
        
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        
        c, r = np.meshgrid(np.arange(x1, x2), np.arange(y1, y2))
        
        z = depth_crop
        x = (c - cx) * z / fx
        y = (r - cy) * z / fy
        
        pts_map = np.stack([x, y, z], axis=-1)
        pts_flat = pts_map.reshape(-1, 3) 

        # --- 4. Generate 'Choose' Indices (Filtered by Depth AND Mask) ---
        # Flatten the arrays to match the flattened 3D points
        z_flat = z.flatten()
        mask_flat = mask_crop.flatten()
        
        # Select pixels where depth is valid AND mask is strictly greater than 0
        valid_indices = np.where((z_flat > 0) & (mask_flat > 0))[0]
        
        # Handle edge cases where the mask might be completely empty
        if len(valid_indices) == 0:
            raise ValueError(f"No valid points found for index {idx} after masking.")
            
        if len(valid_indices) >= self.num_points:
            choose = np.random.choice(valid_indices, self.num_points, replace=False)
        else:
            choose = np.random.choice(valid_indices, self.num_points, replace=True)

        pts = pts_flat[choose, :]

        # --- 5. Process RGB (Cropping and Resizing) ---
        # (Optional but recommended): Apply the mask to the RGB crop to black out background
        rgb_crop = np.array(rgb_full.crop((x1, y1, x2, y2)))
        rgb_crop[mask_crop == 0] = [0, 0, 0] # Black out background pixels
        rgb_crop = Image.fromarray(rgb_crop)
        
        rgb_resized = rgb_crop.resize((self.img_size, self.img_size), Image.BILINEAR)
        rgb_tensor = self.rgb_transform(rgb_resized) 

        # --- 6. Recalculate Choose Indices ---
        rgb_choose = self._get_resize_rgb_choose(choose, bbox, self.img_size)

        return (
            rgb_tensor, 
            torch.from_numpy(pts).float(), 
            torch.from_numpy(rgb_choose).long()
        )

    def __getitem__(self, idx):
        #temp_idx = idx + np.random.randint(-5, 5)
        temp_idx = idx - self.template_offset

        rgb, pts, choose = self._load_data(idx)
        tem_rgb, tem_pts, tem_choose = self._load_data(temp_idx)

        # q_tgt = self.labels["camera_rotation"][idx]
        # q_ref = self.labels["camera_rotation"][temp_idx]


        ret_dict = {
            "rgb": rgb,                   
            "pts": pts,                   
            "choose": choose,             
            "tem1_rgb": tem_rgb,          
            "tem1_pts": tem_pts,          
            "tem1_choose": tem_choose,    
        }

        return ret_dict