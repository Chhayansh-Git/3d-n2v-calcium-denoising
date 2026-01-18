import torch
import tifffile
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path

# --- CONFIGURATION ---
VAL_FILE = r"/Users/chhayanshporwal/Projects/data/val/F0.tif" 
MODEL_PATH = "best_model_n2v.pth"

# --- MODEL (Same as before) ---
class UNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv3d(in_c, out_c, 3, padding=1),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_c, out_c, 3, padding=1),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True)
            )
        self.enc1 = conv_block(1, 32)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = conv_block(32, 64)
        self.pool2 = nn.MaxPool3d(2)
        self.up1 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.dec1 = conv_block(64, 32)
        self.final = nn.Conv3d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        d1 = self.dec1(torch.cat([self.up1(e2), e1], dim=1))
        return self.final(d1)

def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading on {device}...")
    model = UNet3D().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    img = tifffile.imread(VAL_FILE).astype(np.float32)
    
    # Crop Center
    d, h, w = img.shape
    cz, cy, cx = d//2, h//2, w//2
    # Smaller crop, zoom in
    crop = img[cz:cz+8, cy-64:cy+64, cx-64:cx+64] 
    
    # Normalize
    p3, p97 = np.percentile(crop, [3, 97])
    scale = p97 - p3 + 1e-6
    inp = (crop - p3) / scale
    
    tensor = torch.from_numpy(inp).unsqueeze(0).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        pred = model(tensor).cpu().numpy()[0, 0]
    
    pred = (pred * scale) + p3
    
    # --- THE RESIDUAL ---
    # What did the model remove?
    residual = crop - pred
    
    print(f"Max Pixel Value in Original: {crop.max():.2f}")
    print(f"Max Difference (Removed Noise): {np.abs(residual).max():.2f}")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Original
    axes[0].imshow(crop[0], cmap='gray', vmin=p3, vmax=p97)
    axes[0].set_title("Original")
    
    # 2. Denoised
    axes[1].imshow(pred[0], cmap='gray', vmin=p3, vmax=p97)
    axes[1].set_title("Denoised")
    
    # 3. What was Removed (The Noise)
    # We contrast stretch this heavily so you can see the grain
    diff_plot = axes[2].imshow(residual[0], cmap='seismic', vmin=-np.std(residual)*3, vmax=np.std(residual)*3)
    axes[2].set_title("The Removed Noise (Residual)")
    plt.colorbar(diff_plot, ax=axes[2])
    
    plt.show()

if __name__ == "__main__":
    run()