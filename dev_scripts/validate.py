import os
import torch
import tifffile
import numpy as np
import torch.nn as nn
from pathlib import Path

# --- CONFIGURATION (UPDATE THESE PATHS) ---
# Point these to the validation folders on your LOCAL computer
VAL_CLEAN_DIR = r"Path/to/your/validation/clean" 
VAL_NOISY_DIR = r"Path/to/your/validation/noisy"
MODEL_PATH = "best_model_n2v.pth"

# --- 1. METRIC CALCULATION (stSNR) ---
def calculate_snr(gt, pred):
    """Computes basic SNR (Signal to Noise Ratio)"""
    noise = gt - pred
    s_power = np.sum(gt ** 2)
    n_power = np.sum(noise ** 2)
    if n_power == 0: return 100.0
    return 10 * np.log10(s_power / n_power)

def compute_metrics(gt, pred):
    # 1. Spatial SNR (average SNR of each frame)
    s_snr_list = [calculate_snr(gt[t], pred[t]) for t in range(gt.shape[0])]
    sSNR = np.mean(s_snr_list)
    
    # 2. Temporal SNR (average SNR of each pixel over time)
    # Transpose to (H, W, T)
    gt_t = gt.transpose(1, 2, 0)
    pred_t = pred.transpose(1, 2, 0)
    
    noise_t = gt_t - pred_t
    s_power_t = np.sum(gt_t ** 2, axis=2)
    n_power_t = np.sum(noise_t ** 2, axis=2)
    
    # Avoid div by zero
    mask = n_power_t > 0
    t_snr_map = np.zeros_like(s_power_t)
    t_snr_map[mask] = 10 * np.log10(s_power_t[mask] / n_power_t[mask])
    
    tSNR = np.mean(t_snr_map)
    
    # 3. Combined stSNR
    stSNR = 0.5 * sSNR + 0.5 * tSNR
    return stSNR, sSNR, tSNR

# --- 2. MODEL ARCHITECTURE ---
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

# --- 3. RUN VALIDATION ---
def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Validating on {device}...")
    
    # Load Model
    model = UNet3D().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    # Get files
    noisy_files = sorted(list(Path(VAL_NOISY_DIR).glob("*.tif")))
    clean_files = sorted(list(Path(VAL_CLEAN_DIR).glob("*.tif")))
    
    print(f"Found {len(noisy_files)} validation pairs.")
    
    scores = []
    
    for n_path, c_path in zip(noisy_files, clean_files):
        print(f"Processing {n_path.name}...")
        
        # Load Images
        noisy = tifffile.imread(n_path).astype(np.float32)
        clean = tifffile.imread(c_path).astype(np.float32)
        
        # --- SIMPLE CENTER CROP INFERENCE ---
        # We don't need to process the whole massive file just to check if the model works.
        # We take a 32x128x128 chunk from the center.
        d, h, w = noisy.shape
        cz, cy, cx = d//2, h//2, w//2
        
        # Ensure crop is within bounds
        crop_n = noisy[cz:cz+32, cy-64:cy+64, cx-64:cx+64]
        crop_c = clean[cz:cz+32, cy-64:cy+64, cx-64:cx+64]
        
        # Normalize
        p3, p97 = np.percentile(crop_n, [3, 97])
        scale = p97 - p3 + 1e-6
        inp = (crop_n - p3) / scale
        
        # Inference
        tensor = torch.from_numpy(inp).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            pred = model(tensor).cpu().numpy()[0, 0]
            
        # De-normalize
        pred = (pred * scale) + p3
        
        # Calculate Metric
        stSNR, sSNR, tSNR = compute_metrics(crop_c, pred)
        base_stSNR, _, _ = compute_metrics(crop_c, crop_n) # Score before denoising
        
        print(f"  -> Input stSNR: {base_stSNR:.2f} dB")
        print(f"  -> Model stSNR: {stSNR:.2f} dB")
        
        scores.append(stSNR)
        
    print(f"\nFinal Average stSNR: {np.mean(scores):.2f} dB")
    if np.mean(scores) > 12.0: # Arbitrary threshold, but usually good denoising is >12-15dB
        print("✅ SUCCESS: Model is improving quality!")
    else:
        print("⚠️ WARNING: Model might not be learning well.")

if __name__ == "__main__":
    run()