import torch
import tifffile
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path

# --- CONFIGURATION ---
# REPLACE THIS with the actual path to your F0.tif file
VAL_FILE = r"/Users/chhayanshporwal/Projects/data/val/F0.tif" 
MODEL_PATH = "best_model_n2v.pth"

# --- MODEL ARCHITECTURE ---
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
    print(f"Loading model on {device}...")
    
    model = UNet3D().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    print(f"Reading {VAL_FILE}...")
    img = tifffile.imread(VAL_FILE).astype(np.float32)
    
    # Take a center crop to verify (32 frames, 256x256)
    d, h, w = img.shape
    cz, cy, cx = d//2, h//2, w//2
    
    # Crop size
    CD, CH, CW = 16, 128, 128
    
    crop = img[cz:cz+CD, cy-CH:cy+CH, cx-CW:cx+CW]
    
    # Normalize
    p3, p97 = np.percentile(crop, [3, 97])
    scale = p97 - p3 + 1e-6
    inp = (crop - p3) / scale
    
    tensor = torch.from_numpy(inp).unsqueeze(0).unsqueeze(0).float().to(device)
    
    print("Denoising...")
    with torch.no_grad():
        pred = model(tensor).cpu().numpy()[0, 0]
    
    # De-normalize
    pred = (pred * scale) + p3
    
    # --- VISUALIZATION ---
    print("Generating comparison...")
    
    # Pick the middle frame of the crop to show
    mid_frame = CD // 2
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original Noisy
    axes[0].imshow(crop[mid_frame], cmap='gray', vmin=p3, vmax=p97)
    axes[0].set_title("Original (Noisy)")
    axes[0].axis('off')
    
    # Denoised Prediction
    axes[1].imshow(pred[mid_frame], cmap='gray', vmin=p3, vmax=p97)
    axes[1].set_title("Model Output (Denoised)")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\nCHECK THE IMAGE:")
    print("1. Is the 'Model Output' cleaner than 'Original'?")
    print("2. Are the neurons (white blobs) still visible and sharp?")
    print("3. If YES -> SUBMIT IMMEDIATELY.")

if __name__ == "__main__":
    run()