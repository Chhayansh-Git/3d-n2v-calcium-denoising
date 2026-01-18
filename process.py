import os
import torch
import tifffile
import numpy as np
import torch.nn as nn
from pathlib import Path

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
    # --- PATHS (STRICT) ---
    INPUT_PATH = Path("/input/images/calcium-imaging-noisy/")
    OUTPUT_PATH = Path("/output/images/stacked-neuron-images-with-reduced-noise/")
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Model
    model = UNet3D().to(device)
    model.load_state_dict(torch.load("/opt/algorithm/best_model_n2v.pth", map_location=device))
    model.eval()
    
    files = list(INPUT_PATH.glob("*.tif"))
    
    # Inference Config
    patch_size = (32, 128, 128)
    stride = (16, 64, 64)
    
    for f in files:
        img = tifffile.imread(f)
        
        # Pad Image
        d, h, w = img.shape
        pad_d = (patch_size[0] - d % patch_size[0]) % patch_size[0]
        pad_h = (patch_size[1] - h % patch_size[1]) % patch_size[1]
        pad_w = (patch_size[2] - w % patch_size[2]) % patch_size[2]
        img_padded = np.pad(img, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='reflect')
        
        D, H, W = img_padded.shape
        weight_map = np.zeros((D, H, W), dtype=np.float32)
        output_map = np.zeros((D, H, W), dtype=np.float32)
        
        # Sliding Window
        for z in range(0, D - patch_size[0] + 1, stride[0]):
            for y in range(0, H - patch_size[1] + 1, stride[1]):
                for x in range(0, W - patch_size[2] + 1, stride[2]):
                    
                    patch = img_padded[z:z+patch_size[0], y:y+patch_size[1], x:x+patch_size[2]]
                    
                    # Normalize
                    p3, p97 = np.percentile(patch, [3, 97])
                    scale = p97 - p3 + 1e-6
                    patch_norm = (patch - p3) / scale
                    
                    tensor = torch.from_numpy(patch_norm).float().unsqueeze(0).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        pred = model(tensor).cpu().numpy()[0, 0]
                        
                    pred = (pred * scale) + p3
                    
                    output_map[z:z+patch_size[0], y:y+patch_size[1], x:x+patch_size[2]] += pred
                    weight_map[z:z+patch_size[0], y:y+patch_size[1], x:x+patch_size[2]] += 1.0
        
        output_map /= (weight_map + 1e-6)
        final_img = output_map[:d, :h, :w].astype(np.float32)
        
        # SAVE WITH METADATA (MANDATORY)
        tifffile.imwrite(
            OUTPUT_PATH / f.name,
            final_img,
            resolution=(300, 300),
            metadata={'unit': 'um'}
        )

if __name__ == "__main__":
    run()