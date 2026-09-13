import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def vibrance_transform(x: np.ndarray, a: float, sigma: float = 70.0) -> np.ndarray:
    
    x_float = x.astype(np.float64)
    gaussian_boost = a * 128.0 * np.exp(- ((x_float - 128.0) ** 2) / (2.0 * (sigma ** 2)))
    fx = x_float + gaussian_boost
    fx_clipped = np.clip(fx, 0, 255).astype(np.uint8)
    return fx_clipped

# Load Image for Vibrance Enhancement (Fig. 4)
fig4_path = os.path.join("images", "Fig4.jpeg")
bgr_img = cv2.imread(fig4_path)
if bgr_img is None:
    raise FileNotFoundError(f"Could not load image at {fig4_path}")
    
print(f"Loaded {fig4_path} with shape {bgr_img.shape}")

# (a) Split image into Hue, Saturation, and Value planes
hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
H, S, V = cv2.split(hsv_img)

# (b) Apply intensity transformation to Saturation plane with chosen parameter a
# Parameter tuning: a = 0.6 produces vibrant colors while preserving natural tones
a_param = 0.6
sigma_param = 70.0

S_enhanced = vibrance_transform(S, a=a_param, sigma=sigma_param)

# (d) Recombine the three planes
hsv_enhanced = cv2.merge([H, S_enhanced, V])
bgr_enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

rgb_orig = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
rgb_enhanced = cv2.cvtColor(bgr_enhanced, cv2.COLOR_BGR2RGB)

# (e) Display the H, S, V planes, intensity transformation curve, and final comparison

# 1. Save HSV Planes Figure
fig_planes, axes_p = plt.subplots(1, 3, figsize=(15, 5))
axes_p[0].imshow(H, cmap='gray')
axes_p[0].set_title('Hue Plane (H)', fontsize=12, fontweight='bold')
axes_p[0].axis('off')

axes_p[1].imshow(S, cmap='gray')
axes_p[1].set_title('Saturation Plane (S)', fontsize=12, fontweight='bold')
axes_p[1].axis('off')

axes_p[2].imshow(V, cmap='gray')
axes_p[2].set_title('Value Plane (V)', fontsize=12, fontweight='bold')
axes_p[2].axis('off')

plt.tight_layout()
plt.savefig("q4_hsv_planes.png", dpi=300, bbox_inches='tight')
plt.close()

# 2. Main Question 4 Comparison & Transformation Plot
fig = plt.figure(figsize=(14, 10))

# Transformation Curve T(x)
x_vals = np.arange(256, dtype=np.float64)
fx_vals = vibrance_transform(x_vals, a=a_param, sigma=sigma_param)

ax_curve = fig.add_subplot(2, 2, 1)
ax_curve.plot(x_vals, fx_vals, 'r-', linewidth=2.5, label=rf'Vibrance $f(x)$ ($a={a_param}, \sigma={int(sigma_param)}$)')
ax_curve.plot([0, 255], [0, 255], 'k--', alpha=0.4, label='Identity $f(x)=x$')
ax_curve.set_xlim([0, 255]); ax_curve.set_ylim([0, 255])
ax_curve.set_xlabel('Input Saturation ($x$)', fontsize=11)
ax_curve.set_ylabel('Enhanced Saturation ($f(x)$)', fontsize=11)
ax_curve.set_title(f'Vibrance Intensity Transformation ($a = {a_param}$)', fontsize=12, fontweight='bold')
ax_curve.grid(True, linestyle=':', alpha=0.6)
ax_curve.legend(loc='upper left', fontsize=10)

# Enhanced Saturation Plane Comparison
ax_sat = fig.add_subplot(2, 2, 2)
ax_sat.imshow(S_enhanced, cmap='gray')
ax_sat.set_title('Enhanced Saturation Plane ($S_{enhanced}$)', fontsize=12, fontweight='bold')
ax_sat.axis('off')

# Original Image
ax_orig = fig.add_subplot(2, 2, 3)
ax_orig.imshow(rgb_orig)
ax_orig.set_title('Original Image (Fig. 4)', fontsize=12, fontweight='bold')
ax_orig.axis('off')

# Vibrance Enhanced Image
ax_enh = fig.add_subplot(2, 2, 4)
ax_enh.imshow(rgb_enhanced)
ax_enh.set_title(f'Vibrance-Enhanced Image ($a = {a_param}$)', fontsize=12, fontweight='bold')
ax_enh.axis('off')

plt.tight_layout()
output_plot = "q4_result.png"
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
plt.close()
print(f"Question 4 results saved successfully to {output_plot} and q4_hsv_planes.png with a = {a_param}")

