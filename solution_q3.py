import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def gamma_correction_lab(bgr_img: np.ndarray, gamma: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
   
    # 1. Convert BGR to L*a*b* color space
    lab_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab_img)
    
    # 2. Build Gamma Correction Lookup Table for L plane
    r_range = np.arange(256, dtype=np.float64)
    lut_gamma = np.clip(255.0 * ((r_range / 255.0) ** gamma), 0, 255).astype(np.uint8)
    
    # 3. Apply LUT to L plane
    L_corrected = cv2.LUT(L, lut_gamma)
    
    # 4. Merge corrected L plane with original a* and b* planes
    lab_corrected = cv2.merge([L_corrected, a, b])
    
    # 5. Convert back to BGR color space
    bgr_corrected = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)
    
    return bgr_corrected, L, L_corrected


# Load Image for Gamma Correction (Fig. 3)
fig3_path = os.path.join("images", "Fig3.jpeg")
bgr_img = cv2.imread(fig3_path)
if bgr_img is None:
    raise FileNotFoundError(f"Could not load image at {fig3_path}")
    
print(f"Loaded {fig3_path} with shape {bgr_img.shape}")

# Gamma value chosen to brighten dark shadow regions
gamma_val = 0.5
bgr_corrected, L_orig, L_corr = gamma_correction_lab(bgr_img, gamma=gamma_val)

rgb_orig = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
rgb_corrected = cv2.cvtColor(bgr_corrected, cv2.COLOR_BGR2RGB)

# Plotting Q3 Results
fig = plt.figure(figsize=(14, 10))

# 1. Original Image
ax1 = fig.add_subplot(2, 2, 1)
ax1.imshow(rgb_orig)
ax1.set_title('Original Image (Fig. 3)', fontsize=12, fontweight='bold')
ax1.axis('off')

# 2. Gamma Corrected Image
ax2 = fig.add_subplot(2, 2, 2)
ax2.imshow(rgb_corrected)
ax2.set_title(rf'Gamma Corrected Image ($L^*$ plane, $\gamma = {gamma_val}$)', fontsize=12, fontweight='bold')
ax2.axis('off')

# 3. Histograms of L Channel (Original vs Corrected)
ax3 = fig.add_subplot(2, 1, 2)
hist_orig = cv2.calcHist([L_orig], [0], None, [256], [0, 256])
hist_corr = cv2.calcHist([L_corr], [0], None, [256], [0, 256])

ax3.plot(hist_orig, color='black', linewidth=1.8, label=r'Original $L^*$ Plane Histogram')
ax3.plot(hist_corr, color='crimson', linewidth=1.8, label=rf'Gamma Corrected $L^*$ Histogram ($\gamma = {gamma_val}$)')
ax3.set_xlim([0, 255])
ax3.set_xlabel(r'Lightness Intensity ($L^*$ value)', fontsize=11)
ax3.set_ylabel('Pixel Count', fontsize=11)
ax3.set_title(r'Histograms of $L^*$ Plane (Original vs Gamma Corrected)', fontsize=12, fontweight='bold')
ax3.grid(True, linestyle=':', alpha=0.6)
ax3.legend(fontsize=11)

plt.tight_layout()
output_plot = "q3_result.png"
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
plt.close()
print(f"Question 3 results saved successfully to {output_plot} with gamma = {gamma_val}")

