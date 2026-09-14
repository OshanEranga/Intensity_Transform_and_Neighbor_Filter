import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# Load Image for GrabCut (Fig. 8)
fig8_path = os.path.join("images", "Fig8.jpeg")
bgr_img = cv2.imread(fig8_path)
if bgr_img is None:
    raise FileNotFoundError(f"Could not load image at {fig8_path}")
    
H, W = bgr_img.shape[:2]
print(f"Loaded {fig8_path} with shape {W}x{H}")

# -------------------------------------------------------------
# (a) GrabCut Segmentation
# -------------------------------------------------------------
# Define central bounding box rectangle (margin = 10%)
margin_x = int(W * 0.08)
margin_y = int(H * 0.08)
rect = (margin_x, margin_y, W - 2 * margin_x, H - 2 * margin_y)

# Initialize GrabCut mask and models
mask = np.zeros((H, W), dtype=np.uint8)
bgdModel = np.zeros((1, 65), dtype=np.float64)
fgdModel = np.zeros((1, 65), dtype=np.float64)

# Run GrabCut for 5 iterations
cv2.grabCut(bgr_img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

# Generate binary mask: GC_FGD (1) and GC_PR_FGD (3) -> 1, rest -> 0
fg_mask = np.where((mask == 1) | (mask == 3), 1, 0).astype(np.uint8)
bg_mask = 1 - fg_mask

# Extract foreground and background images
foreground = cv2.bitwise_and(bgr_img, bgr_img, mask=fg_mask)
background = cv2.bitwise_and(bgr_img, bgr_img, mask=bg_mask)

# -------------------------------------------------------------
# (b) Produce Enhanced Image with Substantially Blurred Background
# -------------------------------------------------------------
# Apply strong Gaussian blur to full original image
kernel_size = (51, 51)
blurred_background = cv2.GaussianBlur(bgr_img, kernel_size, 0)

# Combine sharp in-focus foreground with blurred background
fg_mask_3ch = np.repeat(fg_mask[:, :, np.newaxis], 3, axis=2)
enhanced_bgr = np.where(fg_mask_3ch == 1, bgr_img, blurred_background)

# Convert images to RGB for Matplotlib visualization
rgb_orig = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
rgb_fg = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)
rgb_bg = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
rgb_enhanced = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)

# -------------------------------------------------------------
# Save Figures for Assignment Report
# -------------------------------------------------------------

# Figure 1: Segmentation Results (Part a)
fig_a, axes_a = plt.subplots(1, 4, figsize=(18, 5))

axes_a[0].imshow(rgb_orig)
axes_a[0].set_title('Original Image (Fig. 8)', fontsize=11, fontweight='bold')
axes_a[0].axis('off')

axes_a[1].imshow(fg_mask * 255, cmap='gray')
axes_a[1].set_title('Final Segmentation Mask', fontsize=11, fontweight='bold')
axes_a[1].axis('off')

axes_a[2].imshow(rgb_fg)
axes_a[2].set_title('Extracted Foreground', fontsize=11, fontweight='bold')
axes_a[2].axis('off')

axes_a[3].imshow(rgb_bg)
axes_a[3].set_title('Extracted Background', fontsize=11, fontweight='bold')
axes_a[3].axis('off')

plt.tight_layout()
plot_a = "q9_segmentation.png"
plt.savefig(plot_a, dpi=300, bbox_inches='tight')
plt.close()

# Figure 2: Original vs Enhanced Comparison (Part b)
fig_b, axes_b = plt.subplots(1, 2, figsize=(14, 7))

axes_b[0].imshow(rgb_orig)
axes_b[0].set_title('Original Image (All In Focus)', fontsize=12, fontweight='bold')
axes_b[0].axis('off')

axes_b[1].imshow(rgb_enhanced)
axes_b[1].set_title('Enhanced Image (Substantially Blurred Background)', fontsize=12, fontweight='bold')
axes_b[1].axis('off')

plt.tight_layout()
plot_b = "q9_result.png"
plt.savefig(plot_b, dpi=300, bbox_inches='tight')
plt.close()

print(f"Question 9 results saved to {plot_a} and {plot_b}")

