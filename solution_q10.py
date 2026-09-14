import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time

def custom_bilateral_filter(img: np.ndarray, d: int, sigma_s: float, sigma_r: float) -> np.ndarray:
   
    img_f = img.astype(np.float64)
    radius = d // 2
    
    # 1. Precompute spatial Gaussian weight matrix W_s (d x d)
    y_grid, x_grid = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1), indexing='ij')
    spatial_dist_sq = x_grid**2 + y_grid**2
    spatial_weights = np.exp(- spatial_dist_sq / (2.0 * (sigma_s ** 2)))
    
    # Pad image to handle boundary conditions
    is_color = (img.ndim == 3)
    if is_color:
        padded = np.pad(img_f, ((radius, radius), (radius, radius), (0, 0)), mode='reflect')
        spatial_weights_broad = spatial_weights[:, :, np.newaxis]
    else:
        padded = np.pad(img_f, ((radius, radius), (radius, radius)), mode='reflect')
        spatial_weights_broad = spatial_weights

    H, W = img.shape[:2]
    output = np.zeros_like(img_f)
    
    # Vectorised neighborhood processing over spatial window shifts
    norm_factor = np.zeros((H, W, 3) if is_color else (H, W), dtype=np.float64)
    
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            # Shifted neighbor patch
            neighbor_patch = padded[radius + dy : radius + dy + H, radius + dx : radius + dx + W]
            
            # Spatial weight for this displacement (dy, dx)
            w_spatial = spatial_weights[dy + radius, dx + radius]
            
            # Range weight based on intensity/color squared difference
            if is_color:
                diff_sq = np.sum((neighbor_patch - img_f) ** 2, axis=2, keepdims=True)
            else:
                diff_sq = (neighbor_patch - img_f) ** 2
                
            w_range = np.exp(- diff_sq / (2.0 * (sigma_r ** 2)))
            
            # Combined bilateral weight
            w_combined = w_spatial * w_range
            
            output += w_combined * neighbor_patch
            norm_factor += w_combined

    # Normalize by total weight
    output = output / (norm_factor + 1e-8)
    return np.clip(np.round(output), 0, 255).astype(np.uint8)

def compute_metrics(im1: np.ndarray, im2: np.ndarray) -> tuple[float, float, float]:
    """
    Computes MSE, PSNR, and MAE between two images.
    """
    diff = im1.astype(np.float64) - im2.astype(np.float64)
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    
    if mse == 0:
        psnr = 100.0
    else:
        psnr = float(10.0 * np.log10((255.0 ** 2) / mse))
        
    return mse, mae, psnr


# Load Image for Bilateral Filtering (Fig. 9)
fig9_path = os.path.join("images", "Fig9.jpeg")
bgr_img = cv2.imread(fig9_path)
if bgr_img is None:
    raise FileNotFoundError(f"Could not load image at {fig9_path}")
    
H, W = bgr_img.shape[:2]
print(f"Loaded {fig9_path} with shape {W}x{H}")

# Parameters Choice
d = 9            # Neighborhood diameter
sigma_s = 5.0    # Spatial Sigma
sigma_r = 45.0   # Range (Color) Sigma

# -------------------------------------------------------------
# (a) OpenCV bilateralFilter vs Gaussian Blur
# -------------------------------------------------------------
t0 = time.time()
cv_gaussian = cv2.GaussianBlur(bgr_img, (d, d), sigmaX=sigma_s, sigmaY=sigma_s)
t_gauss = time.time() - t0

t0 = time.time()
cv_bilateral = cv2.bilateralFilter(bgr_img, d=d, sigmaColor=sigma_r, sigmaSpace=sigma_s)
t_cv_bi = time.time() - t0

# -------------------------------------------------------------
# (b) Custom Bilateral Filter Implementation & Comparison
# -------------------------------------------------------------
t0 = time.time()
custom_bilateral = custom_bilateral_filter(bgr_img, d=d, sigma_s=sigma_s, sigma_r=sigma_r)
t_custom_bi = time.time() - t0

# Quantitative Comparison Metrics
mse_val, mae_val, psnr_val = compute_metrics(custom_bilateral, cv_bilateral)

print(f"==========================================================")
print(f"  BILATERAL FILTER EVALUATION (d={d}, sigma_s={sigma_s}, sigma_r={sigma_r})")
print(f"==========================================================")
print(f"Gaussian Blur Time: {t_gauss*1000:.2f} ms")
print(f"OpenCV Bilateral Time: {t_cv_bi*1000:.2f} ms")
print(f"Custom Bilateral Time: {t_custom_bi*1000:.2f} ms")
print(f"----------------------------------------------------------")
print(f"Quantitative Comparison (Custom vs OpenCV Bilateral):")
print(f"  - Mean Squared Error (MSE): {mse_val:.4f}")
print(f"  - Mean Absolute Error (MAE): {mae_val:.4f} gray levels")
print(f"  - Peak Signal-to-Noise Ratio (PSNR): {psnr_val:.2f} dB")
print(f"==========================================================")

# Convert to RGB for Matplotlib visualization
rgb_orig = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
rgb_gauss = cv2.cvtColor(cv_gaussian, cv2.COLOR_BGR2RGB)
rgb_cv_bi = cv2.cvtColor(cv_bilateral, cv2.COLOR_BGR2RGB)
rgb_custom_bi = cv2.cvtColor(custom_bilateral, cv2.COLOR_BGR2RGB)

# Figure 1: Part (a) OpenCV Bilateral vs Gaussian Blur Comparison
fig_a, axes_a = plt.subplots(1, 3, figsize=(16, 5.5))

axes_a[0].imshow(rgb_orig)
axes_a[0].set_title('Original Image (Fig. 9)', fontsize=11, fontweight='bold')
axes_a[0].axis('off')

axes_a[1].imshow(rgb_gauss)
axes_a[1].set_title(f'Gaussian Blurred (Blurred Edges, $d={d}$)', fontsize=11, fontweight='bold')
axes_a[1].axis('off')

axes_a[2].imshow(rgb_cv_bi)
axes_a[2].set_title(rf'OpenCV Bilateral (Sharp Edges, $\sigma_s={sigma_s}, \sigma_r={sigma_r}$)', fontsize=11, fontweight='bold')
axes_a[2].axis('off')

plt.tight_layout()
plot_a = "q10_part_a.png"
plt.savefig(plot_a, dpi=300, bbox_inches='tight')
plt.close()

# Figure 2: Part (b) Custom vs OpenCV Bilateral Filter Comparison
fig_b, axes_b = plt.subplots(2, 2, figsize=(14, 10))

axes_b[0, 0].imshow(rgb_orig)
axes_b[0, 0].set_title('Original Image (Fig. 9)', fontsize=11, fontweight='bold')
axes_b[0, 0].axis('off')

axes_b[0, 1].imshow(rgb_gauss)
axes_b[0, 1].set_title(f'Gaussian Blurred (Blurred Edges)', fontsize=11, fontweight='bold')
axes_b[0, 1].axis('off')

axes_b[1, 0].imshow(rgb_cv_bi)
axes_b[1, 0].set_title('OpenCV BilateralFilter', fontsize=11, fontweight='bold')
axes_b[1, 0].axis('off')

axes_b[1, 1].imshow(rgb_custom_bi)
axes_b[1, 1].set_title(f'Custom BilateralFilter (PSNR: {psnr_val:.1f} dB)', fontsize=11, fontweight='bold')
axes_b[1, 1].axis('off')

plt.tight_layout()
plot_b = "q10_result.png"
plt.savefig(plot_b, dpi=300, bbox_inches='tight')
plt.close()

print(f"Question 10 plots saved to {plot_a} and {plot_b}")

