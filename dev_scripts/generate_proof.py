import torch
import tifffile
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

# --- PATHS ---
# Use one of your Training or Validation files here
INPUT_FILE = r"/Users/chhayanshporwal/Projects/data/val/F0.tif"  
MODEL_PATH = "best_model_n2v.pth"

# --- MODEL (Must match training) ---
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
    device = "cpu" # CPU is fine for generating one image
    print(f"Loading model...")
    model = UNet3D().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    print(f"Reading {INPUT_FILE}...")
    img = tifffile.imread(INPUT_FILE).astype(np.float32)
    
    # Crop Center to see details
    d, h, w = img.shape
    cz, cy, cx = d//2, h//2, w//2
    crop = img[cz:cz+8, cy-64:cy+64, cx-64:cx+64] # Small 3D crop
    
    # Normalize
    p3, p97 = np.percentile(crop, [3, 97])
    scale = p97 - p3 + 1e-6
    inp = (crop - p3) / scale
    
    tensor = torch.from_numpy(inp).unsqueeze(0).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        pred = model(tensor).cpu().numpy()[0, 0]
    
    pred = (pred * scale) + p3
    residual = crop - pred
    
    # --- PLOT & SAVE ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Use the first frame of the crop
    vmin, vmax = p3, p97
    
    axes[0].imshow(crop[0], cmap='gray', vmin=vmin, vmax=vmax)
    axes[0].set_title("Noisy Input")
    axes[0].axis('off')
    
    axes[1].imshow(pred[0], cmap='gray', vmin=vmin, vmax=vmax)
    axes[1].set_title("Denoised Output")
    axes[1].axis('off')
    
    # Residual (Contrast stretched)
    res_std = np.std(residual)
    diff = axes[2].imshow(residual[0], cmap='seismic', vmin=-res_std*3, vmax=res_std*3)
    axes[2].set_title("Removed Noise (Residual)")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig("proof_of_work.png", dpi=150)
    print("✅ Saved proof_of_work.png")

if __name__ == "__main__":
    run()