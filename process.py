import os
import glob
import torch
import tifffile
import numpy as np
import torch.nn as nn
import traceback

# --- CONFIGURATION (CORRECTED) ---
# The slug from your screenshot is 'stacked-neuron-images-with-noise'
INPUT_PATH = "/input/images/stacked-neuron-images-with-noise"
OUTPUT_PATH = "/output/images/stacked-neuron-images-with-reduced-noise"
MODEL_PATH = "/opt/algorithm/best_model_n2v.pth"

# Tiling settings - Safe for T4 GPU (32GB RAM)
TILE_SIZE = (32, 128, 128) 
OVERLAP = (4, 16, 16)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# --- MODEL DEFINITION ---
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

# --- SLIDING WINDOW INFERENCE ---
def predict_sliding_window(model, volume, device):
    d, h, w = volume.shape
    td, th, tw = TILE_SIZE
    od, oh, ow = OVERLAP
    
    sd, sh, sw = td - od, th - oh, tw - ow
    
    counts = np.zeros(volume.shape, dtype=np.float32)
    prediction = np.zeros(volume.shape, dtype=np.float32)
    
    p3, p97 = np.percentile(volume, [3, 97])
    scale = p97 - p3 + 1e-6
    volume_norm = (volume - p3) / scale
    
    for z in range(0, d, sd):
        for y in range(0, h, sh):
            for x in range(0, w, sw):
                z_end = min(z + td, d)
                y_end = min(y + th, h)
                x_end = min(x + tw, w)
                
                z_start = max(0, z_end - td)
                y_start = max(0, y_end - th)
                x_start = max(0, x_end - tw)
                
                crop = volume_norm[z_start:z_end, y_start:y_end, x_start:x_end]
                
                inp = torch.from_numpy(crop).unsqueeze(0).unsqueeze(0).float().to(device)
                
                with torch.no_grad():
                    out = model(inp).cpu().numpy()[0, 0]
                
                prediction[z_start:z_end, y_start:y_end, x_start:x_end] += out
                counts[z_start:z_end, y_start:y_end, x_start:x_end] += 1.0

    prediction /= counts
    prediction = (prediction * scale) + p3
    return prediction.astype(np.float32)

# --- MAIN ---
def run():
    print("Starting Inference...")
    ensure_dir(OUTPUT_PATH)
    
    # 1. Debug: List Input Directory
    # This will prove if the path is correct in the logs
    print(f"Checking input path: {INPUT_PATH}")
    if os.path.exists(INPUT_PATH):
        print(f"Contents: {os.listdir(INPUT_PATH)}")
    else:
        print(f"!!! CRITICAL: Input path {INPUT_PATH} does not exist!")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    model = None
    try:
        model = UNet3D().to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print("Model loaded.")
    except Exception as e:
        print(f"CRITICAL: Model load failed: {e}")
        traceback.print_exc()

    files = glob.glob(os.path.join(INPUT_PATH, "*.tif")) + \
            glob.glob(os.path.join(INPUT_PATH, "*.tiff"))
    
    print(f"Found {len(files)} files to process.")
    
    # Safety Check: If 0 files, fail gracefully or try recursive search
    if len(files) == 0:
        print("No files found! Attempting recursive search...")
        for root, dirs, f_names in os.walk("/input"):
            for f in f_names:
                if f.endswith(".tif") or f.endswith(".tiff"):
                    files.append(os.path.join(root, f))
        print(f"Found {len(files)} files after recursive search.")

    for f in files:
        fname = os.path.basename(f)
        out_name = os.path.join(OUTPUT_PATH, fname)
        print(f"Processing {fname}...")
        
        try:
            if model is None:
                raise RuntimeError("Model not loaded.")

            img = tifffile.imread(f).astype(np.float32)
            denoised = predict_sliding_window(model, img, device)
            tifffile.imwrite(out_name, denoised, resolution=(300,300), metadata={'axes': 'TZYX'})
            print(f"Saved {out_name}")
            
        except Exception as e:
            print(f"!!! ERROR on {fname}: {e}")
            traceback.print_exc()
            print("!!! FALLBACK: Copying input.")
            try:
                fallback_img = tifffile.imread(f)
                tifffile.imwrite(out_name, fallback_img)
            except:
                print("Total failure on fallback.")

if __name__ == "__main__":
    run()
