
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import os

def intensity_transform(im: np.ndarray, breakpoints: np.ndarray) -> np.ndarray:
    """
    Applies piecewise linear intensity transformation on an input image based on specified breakpoints.
    
    Parameters:
    -----------
    im : np.ndarray
        Input grayscale image (2D numpy array, uint8 or float).
    breakpoints : np.ndarray
        An (N, 2) array/matrix where column 0 represents input intensity values r 
        and column 1 represents corresponding output intensity values s.
        
    Returns:
    --------
    np.ndarray
        Transformed grayscale image of uint8 type with intensity values in [0, 255].
    """
    breakpoints = np.array(breakpoints, dtype=np.float64)
    r_pts = breakpoints[:, 0]
    s_pts = breakpoints[:, 1]
    
    # 256-element lookup table for input intensities r in [0, 255]
    r_range = np.arange(256, dtype=np.float64)
    lut = np.interp(r_range, r_pts, s_pts)
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    
    # Apply LUT mapping to the image
    # If image is float, scale or cast appropriately
    if im.dtype != np.uint8:
        im_uint8 = np.clip(im, 0, 255).astype(np.uint8)
    else:
        im_uint8 = im
        
    transformed_im = lut[im_uint8]
    return transformed_im

def plot_transform_and_images(im, breakpoints_list, titles, save_path="q1_result.png"):
    """
    Plots the intensity transformation curves and side-by-side comparison of original and transformed images.
    """
    num_experiments = len(breakpoints_list)
    fig, axes = plt.subplots(num_experiments, 3, figsize=(14, 4 * num_experiments))
    
    if num_experiments == 1:
        axes = np.expand_dims(axes, axis=0)
        
    for i, (bp, title) in enumerate(zip(breakpoints_list, titles)):
        bp = np.array(bp, dtype=np.float64)
        transformed_im = intensity_transform(im, bp)
        
        # 1. Plot Transformation Curve T(r)
        ax_curve = axes[i, 0]
        r_range = np.arange(256)
        s_range = np.interp(r_range, bp[:, 0], bp[:, 1])
        ax_curve.plot(r_range, s_range, 'r-', linewidth=2, label='Transformation T(r)')
        ax_curve.plot(bp[:, 0], bp[:, 1], 'ko', markersize=6, label='Breakpoints')
        ax_curve.set_xlim([0, 255])
        ax_curve.set_ylim([0, 255])
        ax_curve.set_xlabel('Input Intensity $r$')
        ax_curve.set_ylabel('Output Intensity $s$')
        ax_curve.set_title(f'Intensity Transformation\n({title})')
        ax_curve.grid(True, linestyle='--', alpha=0.6)
        ax_curve.legend(loc='upper left')
        
        # 2. Original Image
        ax_orig = axes[i, 1]
        ax_orig.imshow(im, cmap='gray', vmin=0, vmax=255)
        ax_orig.set_title('Original Image')
        ax_orig.axis('off')
        
        # 3. Transformed Image
        ax_trans = axes[i, 2]
        ax_trans.imshow(transformed_im, cmap='gray', vmin=0, vmax=255)
        ax_trans.set_title(f'Transformed Image\n({title})')
        ax_trans.axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Results saved successfully to {save_path}")

if __name__ == "__main__":
    img_path = os.path.join("images", "Fig1(b).jpeg")
    img = Image.open(img_path).convert('L')
    im_arr = np.array(img)
    print(f"Loaded image {img_path} with shape {im_arr.shape}")

    # Case 1: Example Breakpoints from text
    bp_example = np.array([
        [0, 0],
        [50, 50],
        [100, 150],
        [150, 255],
        [255, 255]
    ])
    
    # Case 2: Exact Breakpoints from Fig. 1a plot
    bp_fig1a = np.array([
        [0, 0],
        [50, 50],
        [50, 100],
        [150, 255],
        [150, 150],
        [255, 255]
    ])
    
    # Case 3: Visually Pleasing Custom Breakpoints (Contrast Stretching)
    # Stretch midtones while preserving shadows and highlights
    bp_pleasing = np.array([
        [0, 0],
        [30, 10],
        [120, 170],
        [220, 245],
        [255, 255]
    ])
    
    breakpoints_list = [bp_example, bp_fig1a, bp_pleasing]
    titles = [
        "Example Breakpoints",
        "Fig. 1a Depicted Transformation",
        "Visually Pleasing Contrast Enhancement"
    ]
    
    plot_transform_and_images(im_arr, breakpoints_list, titles, "q1_comparison.png")
