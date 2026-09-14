import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from typing import Tuple, Dict

def zoom_image(img: np.ndarray, s: float, method: str = 'bilinear') -> np.ndarray:
    
    if not (0 < s <= 10):
        raise ValueError(f"Scale factor s must be in the range (0, 10], got s = {s}")
        
    method_lower = method.lower().strip()
    if method_lower not in ['nearest', 'nearest-neighbor', 'bilinear']:
        raise ValueError(f"Unsupported interpolation method: '{method}'. Choose 'nearest' or 'bilinear'.")

    H, W = img.shape[:2]
    H_out = int(round(H * s))
    W_out = int(round(W * s))
    
    # Destination coordinate grids
    y_out = np.arange(H_out, dtype=np.float64)
    x_out = np.arange(W_out, dtype=np.float64)
    
    # Map output coordinates to input coordinates (pixel-center aligned mapping)
    y_in = (y_out + 0.5) / s - 0.5
    x_in = (x_out + 0.5) / s - 0.5
    
    # Clamp source coordinates to valid image bounds
    y_in = np.clip(y_in, 0, H - 1)
    x_in = np.clip(x_in, 0, W - 1)
    
    if method_lower in ['nearest', 'nearest-neighbor']:
        # Nearest neighbor: round mapped coordinates to closest integer index
        r_in = np.clip(np.floor(y_in + 0.5).astype(int), 0, H - 1)
        c_in = np.clip(np.floor(x_in + 0.5).astype(int), 0, W - 1)
        
        if img.ndim == 2:
            zoomed = img[np.ix_(r_in, c_in)]
        else:
            zoomed = img[r_in[:, None], c_in[None, :]]
            
    else:  # bilinear interpolation
        y0 = np.floor(y_in).astype(int)
        x0 = np.floor(x_in).astype(int)
        
        y1 = np.clip(y0 + 1, 0, H - 1)
        x1 = np.clip(x0 + 1, 0, W - 1)
        
        dy = (y_in - y0)[:, None]  # Shape: (H_out, 1)
        dx = (x_in - x0)[None, :]  # Shape: (1, W_out)
        
        if img.ndim == 3:
            dy = dy[:, :, None]  # Shape: (H_out, 1, 1)
            dx = dx[:, :, None]  # Shape: (1, W_out, 1)
            
        # Sample 4 surrounding pixel values
        I00 = img[y0[:, None], x0[None, :]].astype(np.float64)
        I01 = img[y0[:, None], x1[None, :]].astype(np.float64)
        I10 = img[y1[:, None], x0[None, :]].astype(np.float64)
        I11 = img[y1[:, None], x1[None, :]].astype(np.float64)
        
        # Bilinear combination:
        # I(y,x) = (1-dy)(1-dx)I00 + (1-dy)dx I01 + dy(1-dx)I10 + dy dx I11
        interp = (1 - dy) * ((1 - dx) * I00 + dx * I01) + dy * ((1 - dx) * I10 + dx * I11)
        zoomed = np.clip(np.round(interp), 0, 255).astype(img.dtype)
        
    return zoomed

def compute_metrics(orig: np.ndarray, zoomed: np.ndarray) -> Dict[str, float]:
    
    min_h = min(orig.shape[0], zoomed.shape[0])
    min_w = min(orig.shape[1], zoomed.shape[1])
    
    o_crop = orig[:min_h, :min_w].astype(np.float64)
    z_crop = zoomed[:min_h, :min_w].astype(np.float64)
    
    diff = o_crop - z_crop
    ssd_norm = float(np.sum(diff ** 2) / np.sum(o_crop ** 2))
    mse = float(np.mean(diff ** 2))
    
    if mse == 0:
        psnr = 100.0
    else:
        psnr = float(10.0 * np.log10((255.0 ** 2) / mse))
        
    return {
        'normalized_ssd': ssd_norm,
        'mse': mse,
        'psnr': psnr
    }


image_pairs = [
    ("im01.png", "im01small.png", "Image 1 (im01)"),
    ("im02.png", "im02small.png", "Image 2 (im02)"),
    ("im03.png", "im03small.png", "Image 3 (im03)"),
    ("im04.jpg", "im04small.jpg", "Image 4 (im04)")
]

scale_factor = 4.0
results_summary = []

# Grid visualization setup
fig, axes = plt.subplots(len(image_pairs), 4, figsize=(20, 4.8 * len(image_pairs)))

for idx, (orig_filename, small_filename, label) in enumerate(image_pairs):
    orig_path = os.path.join("images", orig_filename)
    small_path = os.path.join("images", small_filename)
    
    orig_bgr = cv2.imread(orig_path)
    small_bgr = cv2.imread(small_path)
    
    if orig_bgr is None or small_bgr is None:
        print(f"Warning: Could not read image pair ({orig_filename}, {small_filename})")
        continue
        
    print(f"\nProcessing {label}:")
    print(f"  - Original Shape: {orig_bgr.shape[1]}x{orig_bgr.shape[0]}")
    print(f"  - Small Shape   : {small_bgr.shape[1]}x{small_bgr.shape[0]}")
    
    # 1. Zoom using Nearest Neighbor
    t0 = time.time()
    nn_zoomed = zoom_image(small_bgr, s=scale_factor, method='nearest')
    t_nn = time.time() - t0
    metrics_nn = compute_metrics(orig_bgr, nn_zoomed)
    
    # 2. Zoom using Bilinear Interpolation
    t0 = time.time()
    bl_zoomed = zoom_image(small_bgr, s=scale_factor, method='bilinear')
    t_bl = time.time() - t0
    metrics_bl = compute_metrics(orig_bgr, bl_zoomed)
    
    # OpenCV reference benchmark
    cv_bl = cv2.resize(small_bgr, (orig_bgr.shape[1], orig_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    metrics_cv = compute_metrics(orig_bgr, cv_bl)
    
    print(f"  [Nearest Neighbor] SSD Norm: {metrics_nn['normalized_ssd']:.6f} | MSE: {metrics_nn['mse']:.2f} | PSNR: {metrics_nn['psnr']:.2f} dB (Time: {t_nn*1000:.2f}ms)")
    print(f"  [Bilinear Interp ] SSD Norm: {metrics_bl['normalized_ssd']:.6f} | MSE: {metrics_bl['mse']:.2f} | PSNR: {metrics_bl['psnr']:.2f} dB (Time: {t_bl*1000:.2f}ms)")
    print(f"  [OpenCV INTER_LIN] SSD Norm: {metrics_cv['normalized_ssd']:.6f} | MSE: {metrics_cv['mse']:.2f} | PSNR: {metrics_cv['psnr']:.2f} dB")
    
    results_summary.append({
        'label': label,
        'nn': metrics_nn,
        'bl': metrics_bl,
        'cv': metrics_cv
    })
    
    # Convert BGR to RGB for Matplotlib visualization
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    small_rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
    nn_rgb = cv2.cvtColor(nn_zoomed, cv2.COLOR_BGR2RGB)
    bl_rgb = cv2.cvtColor(bl_zoomed, cv2.COLOR_BGR2RGB)
    
    # Row layout: [Small Input, Original Reference, NN Zoomed (4x), Bilinear Zoomed (4x)]
    ax_row = axes[idx]
    
    ax_row[0].imshow(small_rgb)
    ax_row[0].set_title(f"{label} - Input Small\n({small_bgr.shape[1]}x{small_bgr.shape[0]})", fontsize=11)
    ax_row[0].axis('off')
    
    ax_row[1].imshow(orig_rgb)
    ax_row[1].set_title(f"{label} - Ground Truth\n({orig_bgr.shape[1]}x{orig_bgr.shape[0]})", fontsize=11)
    ax_row[1].axis('off')
    
    ax_row[2].imshow(nn_rgb)
    ax_row[2].set_title(f"Nearest-Neighbor (4x)\nNorm SSD: {metrics_nn['normalized_ssd']:.6f} | PSNR: {metrics_nn['psnr']:.2f} dB", fontsize=10, color='darkred')
    ax_row[2].axis('off')
    
    ax_row[3].imshow(bl_rgb)
    ax_row[3].set_title(f"Bilinear Interpolation (4x)\nNorm SSD: {metrics_bl['normalized_ssd']:.6f} | PSNR: {metrics_bl['psnr']:.2f} dB", fontsize=10, color='darkgreen')
    ax_row[3].axis('off')
    
plt.tight_layout()
output_fig_path = "q8_result.png"
plt.savefig(output_fig_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\n[SUCCESS] Visual comparison saved to '{output_fig_path}'")
