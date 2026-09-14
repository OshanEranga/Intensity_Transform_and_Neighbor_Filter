import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time

def custom_conv2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    (b) Custom 2D spatial filtering/correlation implementation.
    """
    img_f = img.astype(np.float64)
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    
    padded = np.pad(img_f, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    output = np.zeros_like(img_f)
    
    # Vectorised 2D sliding correlation
    for m in range(kh):
        for n in range(kw):
            output += kernel[m, n] * padded[m:m + img_f.shape[0], n:n + img_f.shape[1]]
            
    return output

def custom_separable_filter(img: np.ndarray, col_kernel: np.ndarray, row_kernel: np.ndarray) -> np.ndarray:
    """
    (c) Custom 2D filtering using separable 1D row and column kernels.
    Property: Kernel = col_kernel * row_kernel
    """
    img_f = img.astype(np.float64)
    
    # 1. Row filtering with 1D row vector (1 x 3)
    kw = row_kernel.shape[1]
    pad_w = kw // 2
    padded_row = np.pad(img_f, ((0, 0), (pad_w, pad_w)), mode='reflect')
    row_filtered = np.zeros_like(img_f)
    for n in range(kw):
        row_filtered += row_kernel[0, n] * padded_row[:, n:n + img_f.shape[1]]
        
    # 2. Column filtering with 1D column vector (3 x 1)
    kh = col_kernel.shape[0]
    pad_h = kh // 2
    padded_col = np.pad(row_filtered, ((pad_h, pad_h), (0, 0)), mode='reflect')
    output = np.zeros_like(img_f)
    for m in range(kh):
        output += col_kernel[m, 0] * padded_col[m:m + img_f.shape[0], :]
        
    return output


# Load Image for Sobel Filtering (Fig. 7)
fig7_path = os.path.join("images", "Fig7.jpeg")
img_gray = cv2.imread(fig7_path, cv2.IMREAD_GRAYSCALE)
if img_gray is None:
    raise FileNotFoundError(f"Could not load image at {fig7_path}")
    
print(f"Loaded {fig7_path} with shape {img_gray.shape}")

# Define Sobel Kernels as per prompt specification
# Kx = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
Kx = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
], dtype=np.float64)

# Ky = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
Ky = np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]
], dtype=np.float64)

# -------------------------------------------------------------
# (a) Using existing cv2.filter2D
# -------------------------------------------------------------
t0 = time.time()
gx_a = cv2.filter2D(img_gray, cv2.CV_64F, Kx)
gy_a = cv2.filter2D(img_gray, cv2.CV_64F, Ky)
mag_a = np.sqrt(gx_a**2 + gy_a**2)
t_a = time.time() - t0

# -------------------------------------------------------------
# (b) Writing custom code for Sobel filtering
# -------------------------------------------------------------
t0 = time.time()
gx_b = custom_conv2d(img_gray, Kx)
gy_b = custom_conv2d(img_gray, Ky)
mag_b = np.sqrt(gx_b**2 + gy_b**2)
t_b = time.time() - t0

# -------------------------------------------------------------
# (c) Using separable property:
# Kx = [1, 2, 1]^T * [1, 0, -1]
# Ky = [1, 0, -1]^T * [1, 2, 1]
# -------------------------------------------------------------
col_x = np.array([[1], [2], [1]], dtype=np.float64)
row_x = np.array([[1, 0, -1]], dtype=np.float64)

col_y = np.array([[1], [0], [-1]], dtype=np.float64)
row_y = np.array([[1, 2, 1]], dtype=np.float64)

t0 = time.time()
gx_c = custom_separable_filter(img_gray, col_x, row_x)
gy_c = custom_separable_filter(img_gray, col_y, row_y)
mag_c = np.sqrt(gx_c**2 + gy_c**2)
t_c = time.time() - t0

# Validation
diff_ab = np.max(np.abs(mag_a - mag_b))
diff_ac = np.max(np.abs(mag_a - mag_c))

print(f"Method (a) filter2D Time: {t_a*1000:.2f} ms")
print(f"Method (b) Custom 2D Time: {t_b*1000:.2f} ms")
print(f"Method (c) Separable 1D Time: {t_c*1000:.2f} ms")
print(f"Max absolute error (a vs b): {diff_ab:.6e}")
print(f"Max absolute error (a vs c): {diff_ac:.6e}")

# Plotting Results
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

# Row 1: Method (a) cv2.filter2D
axes[0, 0].imshow(np.abs(gx_a), cmap='gray'); axes[0, 0].set_title('(a) filter2D $G_x$', fontsize=11, fontweight='bold'); axes[0, 0].axis('off')
axes[0, 1].imshow(np.abs(gy_a), cmap='gray'); axes[0, 1].set_title('(a) filter2D $G_y$', fontsize=11, fontweight='bold'); axes[0, 1].axis('off')
axes[0, 2].imshow(mag_a, cmap='gray'); axes[0, 2].set_title('(a) filter2D Gradient Magnitude', fontsize=11, fontweight='bold'); axes[0, 2].axis('off')

# Row 2: Method (b) Custom 2D
axes[1, 0].imshow(np.abs(gx_b), cmap='gray'); axes[1, 0].set_title('(b) Custom 2D $G_x$', fontsize=11, fontweight='bold'); axes[1, 0].axis('off')
axes[1, 1].imshow(np.abs(gy_b), cmap='gray'); axes[1, 1].set_title('(b) Custom 2D $G_y$', fontsize=11, fontweight='bold'); axes[1, 1].axis('off')
axes[1, 2].imshow(mag_b, cmap='gray'); axes[1, 2].set_title('(b) Custom 2D Gradient Magnitude', fontsize=11, fontweight='bold'); axes[1, 2].axis('off')

# Row 3: Method (c) Separable 1D
axes[2, 0].imshow(np.abs(gx_c), cmap='gray'); axes[2, 0].set_title('(c) Separable 1D $G_x$', fontsize=11, fontweight='bold'); axes[2, 0].axis('off')
axes[2, 1].imshow(np.abs(gy_c), cmap='gray'); axes[2, 1].set_title('(c) Separable 1D $G_y$', fontsize=11, fontweight='bold'); axes[2, 1].axis('off')
axes[2, 2].imshow(mag_c, cmap='gray'); axes[2, 2].set_title('(c) Separable 1D Gradient Magnitude', fontsize=11, fontweight='bold'); axes[2, 2].axis('off')

plt.tight_layout()
output_plot = "q7_result.png"
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
plt.close()
print(f"Question 7 results saved successfully to {output_plot}")

