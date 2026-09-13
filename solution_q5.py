import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def histogram_equalization(im: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    im_uint8 = np.clip(im, 0, 255).astype(np.uint8)
    num_pixels = im_uint8.size
    
    # 1. Compute original histogram (pixel counts for intensities 0 to 255)
    hist_before = np.bincount(im_uint8.ravel(), minlength=256)
    
    # 2. Compute Probability Mass Function (PMF)
    pmf = hist_before / num_pixels
    
    # 3. Compute Cumulative Distribution Function (CDF)
    cdf = np.cumsum(pmf)
    
    # 4. Equalization Transformation Function: T(k) = round((L - 1) * CDF(k))
    lut = np.round(255.0 * cdf).astype(np.uint8)
    
    # 5. Apply Transformation LUT to map image pixels
    equalized_im = lut[im_uint8]
    
    # 6. Compute histogram after equalization
    hist_after = np.bincount(equalized_im.ravel(), minlength=256)
    
    return equalized_im, hist_before, hist_after, cdf


# Load Image for Histogram Equalization (Fig. 5)
fig5_path = os.path.join("images", "Fig5.tif")
img_gray = cv2.imread(fig5_path, cv2.IMREAD_GRAYSCALE)
if img_gray is None:
    raise FileNotFoundError(f"Could not load image at {fig5_path}")
    
print(f"Loaded {fig5_path} with shape {img_gray.shape}")

# Perform Custom Histogram Equalization
eq_img, hist_before, hist_after, cdf = histogram_equalization(img_gray)

# Create Visualization Figure
fig = plt.figure(figsize=(15, 10))

# 1. Original Image
ax1 = fig.add_subplot(2, 2, 1)
ax1.imshow(img_gray, cmap='gray', vmin=0, vmax=255)
ax1.set_title('Original Image (Fig. 5)', fontsize=12, fontweight='bold')
ax1.axis('off')

# 2. Equalized Image
ax2 = fig.add_subplot(2, 2, 2)
ax2.imshow(eq_img, cmap='gray', vmin=0, vmax=255)
ax2.set_title('Custom Histogram Equalized Image', fontsize=12, fontweight='bold')
ax2.axis('off')

# 3. Histogram Before Equalization + CDF
ax3 = fig.add_subplot(2, 2, 3)
r_vals = np.arange(256)
ax3.bar(r_vals, hist_before, color='navy', width=1.0, alpha=0.7, label='Histogram (Before)')
ax3.set_xlim([0, 255])
ax3.set_xlabel('Pixel Intensity Level', fontsize=11)
ax3.set_ylabel('Pixel Count (Frequency)', fontsize=11, color='navy')
ax3.set_title('Histogram Before Equalization', fontsize=12, fontweight='bold')
ax3.grid(True, linestyle=':', alpha=0.5)

# Secondary Y-axis for CDF
ax3_cdf = ax3.twinx()
ax3_cdf.plot(r_vals, cdf, color='crimson', linewidth=2.0, label='CDF')
ax3_cdf.set_ylabel('Normalized CDF', fontsize=11, color='crimson')
ax3_cdf.set_ylim([0, 1.05])

# Combine legends
lines_1, labels_1 = ax3.get_legend_handles_labels()
lines_2, labels_2 = ax3_cdf.get_legend_handles_labels()
ax3.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=10)

# 4. Histogram After Equalization
ax4 = fig.add_subplot(2, 2, 4)
ax4.bar(r_vals, hist_after, color='darkgreen', width=1.0, alpha=0.7, label='Histogram (After)')
ax4.set_xlim([0, 255])
ax4.set_xlabel('Pixel Intensity Level', fontsize=11)
ax4.set_ylabel('Pixel Count (Frequency)', fontsize=11)
ax4.set_title('Histogram After Equalization', fontsize=12, fontweight='bold')
ax4.grid(True, linestyle=':', alpha=0.5)
ax4.legend(loc='upper left', fontsize=10)

plt.tight_layout()
output_plot = "q5_result.png"
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
plt.close()
print(f"Question 5 results saved successfully to {output_plot}")
