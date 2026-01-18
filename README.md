# AI4Life 2025: Unsupervised 3D Calcium Imaging Denoising

This repository contains the solution for the **AI4Life Calcium Imaging Denoising Challenge (CIDC 2025)**. The goal is to remove noise from 3D calcium imaging sequences while preserving the temporal dynamics and spatial structure of neurons.

## 🧠 Approach: 3D Noise2Void (Unsupervised)

Since ground-truth (clean) data was not available for training (per challenge rules), this solution utilizes **Noise2Void (N2V)**, a self-supervised training scheme.

### Key Features:
* **3D U-Net Architecture:** Processes video as a 3D volume `(Time, Height, Width)` to ensure temporal consistency (`tSNR`).
* **Blind-Spot Training:** The network learns to predict the intensity of a masked pixel based *only* on its spatial and temporal neighbors, effectively learning to suppress independent noise without needing clean targets.
* **Sliding Window Inference:** Handles large input volumes by processing overlapping 3D patches to prevent edge artifacts.
* **Robust Normalization:** Uses 3rd-97th percentile scaling to handle intensity outliers common in microscopy data.

## 📊 Performance & Proof of Work

The model was validated on a held-out local validation set. The residual analysis confirms that the model removes high-frequency noise (grain) without removing biological structures (neurons).

![Denoising Results](proof_of_work.png)

* **Left:** Noisy Input (Raw Microscopy)
* **Center:** Denoised Output (Smoother, neurons preserved)
* **Right:** Residual (Shows only random noise was removed, no structural leakage)

## 🛠️ Repository Structure

* `Dockerfile`: Defines the reproducible container environment (Python 3.9, PyTorch).
* `process.py`: Main inference script. Handles I/O, tiling, and metadata requirements.
* `best_model_n2v.pth`: Trained model weights (Epoch 50).
* `requirements.txt`: Project dependencies.

## 🚀 Usage (Docker)

To build and run this container locally:

```bash
# Build
docker build -t calcium-denoiser .

# Run (Mounting input/output directories)
docker run --rm \
    -v /path/to/local/input:/input \
    -v /path/to/local/output:/output \
    calcium-denoiser

```
