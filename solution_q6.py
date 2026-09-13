import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# 1. Load Image Fig. 6
fig6_path = os.path.join("images", "Fig6.jpeg")
bgr_img = cv2.imread(fig6_path)
if bgr_img is None:
    raise FileNotFoundError(f"Could not load image at {fig6_path}")
    
print(f"Loaded {fig6_path} with shape {bgr_img.shape}")

# (a) Convert to HSV and split into Hue, Saturation, and Value planes
hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
H, S, V = cv2.split(hsv_img)

# (b) Threshold the Saturation plane to extract binary foreground mask
# Neutral gray background has near zero saturation (S < 15)
thresh_val = 15
_, mask_raw = cv2.threshold(S, thresh_val, 255, cv2.THRESH_BINARY)

# Clean mask using morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
mask = cv2.morphologyEx(mask_raw, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# (c) Obtain foreground only using cv.bitwise_and and compute histogram
foreground_V = cv2.bitwise_and(V, V, mask=mask)

# Extract foreground pixels only (where mask > 0)
fg_pixels = V[mask > 0]
num_fg_pixels = fg_pixels.size

# Compute histogram of foreground
hist_fg = np.bincount(fg_pixels, minlength=256)

# (d) Obtain the cumulative sum of the histogram using np.cumsum
cdf_fg = np.cumsum(hist_fg)

# (e) Histogram-equalize the foreground using standard formula
# T(k) = round( (L - 1) * (cdf(k) - cdf_min) / (N_fg - cdf_min) )
cdf_min = cdf_fg[cdf_fg > 0].min() if np.any(cdf_fg > 0) else 0
lut_fg = np.round(255.0 * (cdf_fg - cdf_min) / (num_fg_pixels - cdf_min + 1e-6))
lut_fg = np.clip(lut_fg, 0, 255).astype(np.uint8)

# Apply lookup table to V channel for foreground pixels
V_eq = lut_fg[V]
V_fg_eq = cv2.bitwise_and(V_eq, V_eq, mask=mask)

# (f) Extract background and add with histogram-equalized foreground
mask_bg = cv2.bitwise_not(mask)
V_bg = cv2.bitwise_and(V, V, mask=mask_bg)

# Add background V and equalized foreground V
V_final = cv2.add(V_bg, V_fg_eq)

# Recombine planes and convert to BGR/RGB
hsv_final = cv2.merge([H, S, V_final])
bgr_final = cv2.cvtColor(hsv_final, cv2.COLOR_HSV2BGR)

rgb_orig = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
rgb_final = cv2.cvtColor(bgr_final, cv2.COLOR_BGR2RGB)

# --- Visualization ---
# Show Hue, Saturation, Value planes, Binary Mask, Original Image, and Equalized Result
fig = plt.figure(figsize=(16, 11))

# 1. Hue Plane
ax1 = fig.add_subplot(2, 3, 1)
ax1.imshow(H, cmap='gray')
ax1.set_title('Hue Plane (H)', fontsize=12, fontweight='bold')
ax1.axis('off')

# 2. Saturation Plane
ax2 = fig.add_subplot(2, 3, 2)
ax2.imshow(S, cmap='gray')
ax2.set_title('Saturation Plane (S)', fontsize=12, fontweight='bold')
ax2.axis('off')

# 3. Value Plane
ax3 = fig.add_subplot(2, 3, 3)
ax3.imshow(V, cmap='gray')
ax3.set_title('Value Plane (V)', fontsize=12, fontweight='bold')
ax3.axis('off')

# 4. Foreground Mask
ax4 = fig.add_subplot(2, 3, 4)
ax4.imshow(mask, cmap='gray')
ax4.set_title('Foreground Mask (S > 15)', fontsize=12, fontweight='bold')
ax4.axis('off')

# 5. Original Image
ax5 = fig.add_subplot(2, 3, 5)
ax5.imshow(rgb_orig)
ax5.set_title('Original Image (Fig. 6)', fontsize=12, fontweight='bold')
ax5.axis('off')

# 6. Foreground Equalized Result
ax6 = fig.add_subplot(2, 3, 6)
ax6.imshow(rgb_final)
ax6.set_title('Foreground Equalized Result', fontsize=12, fontweight='bold')
ax6.axis('off')

plt.tight_layout()
output_plot = "q6_result.png"
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
plt.close()
print(f"Question 6 results saved successfully to {output_plot}")


