import matplotlib.pyplot as plt
import numpy as np
from database import PanoptesDataset

def debug_visualize(dataset, idx=0):
    """
    Visualizes the RGB crop, 3D point cloud, and 2D-3D alignment from the dataset.
    """
    # 1. Fetch the dictionary from the dataset
    ret_dict = dataset[idx]
    
    # Extract tensors and convert them to NumPy arrays
    rgb_tensor = ret_dict["rgb"]       # Shape: [3, 224, 224]
    pts = ret_dict["pts"].numpy()      # Shape: [1024, 3]
    choose = ret_dict["choose"].numpy() # Shape: [1024]
    
    # 2. Format the RGB image
    # PyTorch tensors are [Channels, Height, Width]. Matplotlib needs [Height, Width, Channels].
    rgb = rgb_tensor.permute(1, 2, 0).numpy()
    img_size = rgb.shape[0] # Usually 224
    
    # 3. Map the 1D 'choose' indices back to 2D (x, y) coordinates on the 224x224 image
    y_coords = choose // img_size
    x_coords = choose % img_size
    
    # Extract the Z-coordinate (depth) to use for color-grading
    depths = pts[:, 2] 
    
    # --- Plotting Setup ---
    fig = plt.figure(figsize=(18, 6))
    
    # Plot 1: Cropped & Resized RGB Image
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(rgb)
    ax1.set_title("1. Resized RGB Crop")
    ax1.axis("off")
    
    # Plot 2: 3D Point Cloud
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    # We scatter X, Y, Z. Inverting Y and Z is sometimes needed depending on the camera frame, 
    # but we plot raw coordinates here.
    p3d = ax2.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=depths, cmap='jet', s=5)
    ax2.set_title("2. 3D Point Cloud")
    ax2.set_xlabel("X (meters)")
    ax2.set_ylabel("Y (meters)")
    ax2.set_zlabel("Z / Depth (meters)")
    
    # Plot 3: 3D Points overlaid on the 2D RGB Image
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(rgb)
    # Scatter the calculated (x, y) coordinates over the image
    p2d = ax3.scatter(x_coords, y_coords, c=depths, cmap='jet', s=10, alpha=0.9, edgecolors='none')
    ax3.set_title("3. Points on RGB (Color = Depth)")
    ax3.axis("off")
    
    # Add a colorbar for the depth scale
    cbar = plt.colorbar(p2d, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label("Depth (meters)")
    
    plt.tight_layout()
    plt.show()

# To use it, simply instantiate your dataset and call the function:
my_dataset = PanoptesDataset(dataset_path="./panoptes-datasets/integral")
debug_visualize(my_dataset, idx=0)