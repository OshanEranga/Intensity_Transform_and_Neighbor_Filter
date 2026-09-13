import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def intensity_transform(im: np.ndarray, breakpoints: np.ndarray) -> np.ndarray:
    
    breakpoints = np.array(breakpoints, dtype=np.float64)
    r_pts = breakpoints[:, 0]
    s_pts = breakpoints[:, 1]
    
    r_range = np.arange(256, dtype=np.float64)
    lut = np.interp(r_range, r_pts, s_pts)
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    
    im_uint8 = np.clip(im, 0, 255).astype(np.uint8) if im.dtype != np.uint8 else im
    return lut[im_uint8]


# Load Brain Proton Density Image (Fig. 2)
fig2_path = os.path.join("images", "Fig2.jpeg")
im_brain = cv2.imread(fig2_path, cv2.IMREAD_GRAYSCALE)
if im_brain is None:
    raise FileNotFoundError(f"Could not load image at {fig2_path}")
    
print(f"Loaded {fig2_path} with shape {im_brain.shape}")

# 1. Breakpoints for White Matter (WM) Accentuation
# White matter corresponds to high intensities (approx 130-255).
# Suppress low/mid intensities, stretch high intensities to 255.
bp_wm = np.array([
    [0, 0],
    [130, 10],
    [170, 180],
    [230, 255],
    [255, 255]
])

# 2. Breakpoints for Gray Matter (GM) Accentuation
# Gray matter corresponds to intermediate intensities (approx 70-160).
# Bandpass transformation: boost mid intensities, suppress low (CSF) and high (WM).
bp_gm = np.array([
    [0, 0],
    [70, 10],
    [110, 255],
    [150, 255],
    [185, 20],
    [255, 0]
])

im_wm = intensity_transform(im_brain, bp_wm)
im_gm = intensity_transform(im_brain, bp_gm)

# Plotting Q2 Results
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
r_range = np.arange(256)

# Row 1: White Matter
s_wm = np.interp(r_range, bp_wm[:, 0], bp_wm[:, 1])
axes[0, 0].plot(r_range, s_wm, 'r-', linewidth=2.5, label=r'$T_{WM}(r)$')
axes[0, 0].scatter(bp_wm[:, 0], bp_wm[:, 1], color='black', zorder=5, s=40)
axes[0, 0].set_title('Intensity Transformation for White Matter', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Input Intensity ($r$)', fontsize=11)
axes[0, 0].set_ylabel('Output Intensity ($s$)', fontsize=11)
axes[0, 0].set_xlim([0, 255]); axes[0, 0].set_ylim([0, 255])
axes[0, 0].grid(True, linestyle=':', alpha=0.6)
axes[0, 0].legend(loc='upper left')

axes[0, 1].imshow(im_brain, cmap='gray', vmin=0, vmax=255)
axes[0, 1].set_title('Original Brain PD Image (Fig. 2)', fontsize=12, fontweight='bold')
axes[0, 1].axis('off')

axes[0, 2].imshow(im_wm, cmap='gray', vmin=0, vmax=255)
axes[0, 2].set_title('Accentuated White Matter', fontsize=12, fontweight='bold')
axes[0, 2].axis('off')

# Row 2: Gray Matter
s_gm = np.interp(r_range, bp_gm[:, 0], bp_gm[:, 1])
axes[1, 0].plot(r_range, s_gm, 'b-', linewidth=2.5, label=r'$T_{GM}(r)$')
axes[1, 0].scatter(bp_gm[:, 0], bp_gm[:, 1], color='black', zorder=5, s=40)
axes[1, 0].set_title('Intensity Transformation for Gray Matter', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Input Intensity ($r$)', fontsize=11)
axes[1, 0].set_ylabel('Output Intensity ($s$)', fontsize=11)
axes[1, 0].set_xlim([0, 255]); axes[1, 0].set_ylim([0, 255])
axes[1, 0].grid(True, linestyle=':', alpha=0.6)
axes[1, 0].legend(loc='upper left')

axes[1, 1].imshow(im_brain, cmap='gray', vmin=0, vmax=255)
axes[1, 1].set_title('Original Brain PD Image (Fig. 2)', fontsize=12, fontweight='bold')
axes[1, 1].axis('off')

axes[1, 2].imshow(im_gm, cmap='gray', vmin=0, vmax=255)
axes[1, 2].set_title('Accentuated Gray Matter', fontsize=12, fontweight='bold')
axes[1, 2].axis('off')

plt.tight_layout()
output_plot = "q2_result.png"
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
plt.close()
print(f"Question 2 results saved successfully to {output_plot}")

