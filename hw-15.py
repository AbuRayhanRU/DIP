import numpy as np
import matplotlib.pyplot as plt
import cv2
from numpy.fft import fft2, fftshift, ifftshift, ifft2


def create_filter_grid(rows, cols):
    center_r, center_c = rows // 2, cols // 2
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    distance = np.sqrt((c - center_c)**2 + (r - center_r)**2)
    return distance

def create_ideal_filter(shape, cutoff, filter_type='lp'):
    distance = create_filter_grid(shape[0], shape[1])
    if filter_type == 'lp':
        mask = distance <= cutoff
    elif filter_type == 'hp':
        mask = distance > cutoff
    elif filter_type == 'bp':
        cutoff_high = cutoff[1]
        cutoff_low = cutoff[0]
        mask = (distance >= cutoff_low) & (distance <= cutoff_high)
    return mask.astype(float)

def create_gaussian_filter(shape, cutoff, filter_type='lp'):
    distance = create_filter_grid(shape[0], shape[1])
    if filter_type == 'lp':
        mask = np.exp(-(distance**2) / (2 * cutoff**2))
    elif filter_type == 'hp':
        mask = 1 - np.exp(-(distance**2) / (2 * cutoff**2))
    elif filter_type == 'bp':
        cutoff_high = cutoff[1]
        cutoff_low = cutoff[0]
        lp_high = np.exp(-(distance**2) / (2 * cutoff_high**2))
        lp_low = np.exp(-(distance**2) / (2 * cutoff_low**2))
        mask = lp_high - lp_low
    return mask

def create_butterworth_filter(shape, cutoff, order_n, filter_type='lp'):
    distance = create_filter_grid(shape[0], shape[1])
    distance = distance + 1e-6 # Avoid division by zero
    if filter_type == 'lp':
        mask = 1 / (1 + (distance / cutoff)**(2 * order_n))
    elif filter_type == 'hp':
        mask = 1 / (1 + (cutoff / distance)**(2 * order_n))
    elif filter_type == 'bp':
        cutoff_high = cutoff[1]
        cutoff_low = cutoff[0]
        lp_high = 1 / (1 + (distance / cutoff_high)**(2 * order_n))
        lp_low = 1 / (1 + (distance / cutoff_low)**(2 * order_n))
        mask = lp_high - lp_low
    return mask

def apply_frequency_filter(image, filter_mask):
    f_transform = fft2(image)
    f_transform_shifted = fftshift(f_transform)
    
    f_transform_filtered = f_transform_shifted * filter_mask
    
    f_transform_uncentered = ifftshift(f_transform_filtered)
    img_filtered_complex = ifft2(f_transform_uncentered)
    
    img_filtered = np.abs(img_filtered_complex)
    
    img_filtered = 255 * (img_filtered - np.min(img_filtered)) / \
                   (np.max(img_filtered) - np.min(img_filtered))
    
    return img_filtered.astype(np.uint8)

if __name__ == "__main__":
    img_path = 'images/image2.jpg' 
    
    img_normal = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    min_val, max_val, _, _ = cv2.minMaxLoc(img_normal)
    target_min, target_max = 50, 150
    alpha = (target_max - target_min) / (max_val - min_val)
    beta = target_min - min_val * alpha
    img_low = cv2.convertScaleAbs(img_normal, alpha=alpha, beta=beta)

    img_high = cv2.equalizeHist(img_normal)

    images = {
        "Low Contrast": img_low,
        "Normal Contrast": img_normal,
        "High Contrast": img_high
    }
    
    CUTOFF_LP = 40
    CUTOFF_HP = 40
    CUTOFF_BP = (30, 60)
    BUTTERWORTH_N = 2
    IMG_SHAPE = img_normal.shape

    gauss_lp = create_gaussian_filter(IMG_SHAPE, CUTOFF_LP, 'lp')
    gauss_hp = create_gaussian_filter(IMG_SHAPE, CUTOFF_HP, 'hp')
    gauss_bp = create_gaussian_filter(IMG_SHAPE, CUTOFF_BP, 'bp')
    
    butter_lp = create_butterworth_filter(IMG_SHAPE, CUTOFF_LP, BUTTERWORTH_N, 'lp')
    butter_hp = create_butterworth_filter(IMG_SHAPE, CUTOFF_HP, BUTTERWORTH_N, 'hp')
    butter_bp = create_butterworth_filter(IMG_SHAPE, CUTOFF_BP, BUTTERWORTH_N, 'bp')
    
    ideal_lp = create_ideal_filter(IMG_SHAPE, CUTOFF_LP, 'lp')
    ideal_hp = create_ideal_filter(IMG_SHAPE, CUTOFF_HP, 'hp')
    ideal_bp = create_ideal_filter(IMG_SHAPE, CUTOFF_BP, 'bp')

    
    # Plot 1: Gaussian Filtering on Varying Contrast
    fig1, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig1.suptitle("Figure 1: Application of Gaussian Filters on Varying Contrast Images", 
                  fontsize=20, y=1.02)
    
    for i, (name, img) in enumerate(images.items()):
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f"Original ({name})", fontsize=12)
        
        filtered_lp = apply_frequency_filter(img, gauss_lp)
        axes[i, 1].imshow(filtered_lp, cmap='gray')
        axes[i, 1].set_title(f"Gaussian LPF (D0={CUTOFF_LP})", fontsize=12)
        
        filtered_hp = apply_frequency_filter(img, gauss_hp)
        axes[i, 2].imshow(filtered_hp, cmap='gray')
        axes[i, 2].set_title(f"Gaussian HPF (D0={CUTOFF_HP})", fontsize=12)

        filtered_bp = apply_frequency_filter(img, gauss_bp)
        axes[i, 3].imshow(filtered_bp, cmap='gray')
        axes[i, 3].set_title(f"Gaussian BPF (D0={CUTOFF_BP})", fontsize=12)

    for ax in axes.flat:
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig("figure_1_gaussian_filtering.png")
    plt.show()

    # Plot 2: Butterworth Filtering on Varying Contrast
    fig2, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig2.suptitle(f"Figure 2: Application of Butterworth Filters (n={BUTTERWORTH_N}) on Varying Contrast", 
                  fontsize=20, y=1.02)
    
    for i, (name, img) in enumerate(images.items()):
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f"Original ({name})", fontsize=12)
        
        filtered_lp = apply_frequency_filter(img, butter_lp)
        axes[i, 1].imshow(filtered_lp, cmap='gray')
        axes[i, 1].set_title(f"Butterworth LPF (n={BUTTERWORTH_N})", fontsize=12)
        
        filtered_hp = apply_frequency_filter(img, butter_hp)
        axes[i, 2].imshow(filtered_hp, cmap='gray')
        axes[i, 2].set_title(f"Butterworth HPF (n={BUTTERWORTH_N})", fontsize=12)

        filtered_bp = apply_frequency_filter(img, butter_bp)
        axes[i, 3].imshow(filtered_bp, cmap='gray')
        axes[i, 3].set_title(f"Butterworth BPF (n={BUTTERWORTH_N})", fontsize=12)

    for ax in axes.flat:
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig("figure_2_butterworth_filtering.png")
    plt.show()

    # Plot 3: Comparative Analysis (Ideal, Gaussian, Butterworth)
    fig3, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig3.suptitle("Figure 3: Comparison of Ideal, Gaussian, and Butterworth (n=2) Filters", 
                  fontsize=20, y=1.02)
    
    img_to_compare = images["Normal Contrast"]
    
    filters_to_compare = {
        "Ideal": (ideal_lp, ideal_hp),
        "Gaussian": (gauss_lp, gauss_hp),
        "Butterworth (n=2)": (butter_lp, butter_hp)
    }
    
    for i, (name, (lp_mask, hp_mask)) in enumerate(filters_to_compare.items()):
        axes[i, 0].imshow(img_to_compare, cmap='gray')
        axes[i, 0].set_title(f"Original (for {name})", fontsize=12)
        
        filtered_lp = apply_frequency_filter(img_to_compare, lp_mask)
        axes[i, 1].imshow(filtered_lp, cmap='gray')
        axes[i, 1].set_title(f"{name} LPF (D0={CUTOFF_LP})", fontsize=12)
        
        filtered_hp = apply_frequency_filter(img_to_compare, hp_mask)
        axes[i, 2].imshow(filtered_hp, cmap='gray')
        axes[i, 2].set_title(f"{name} HPF (D0={CUTOFF_HP})", fontsize=12)

    for ax in axes.flat:
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig("figure_3_filter_comparison.png")
    plt.show()


    # Plot 4: Effect of Order 'n' in Butterworth Filtering
    orders = [1, 2, 5, 10]
    fig4, axes = plt.subplots(len(orders), 3, figsize=(15, 20))
    fig4.suptitle("Figure 4: Effect of Varying Order 'n' on Butterworth Filtering", 
                  fontsize=20, y=1.01)

    for i, n in enumerate(orders):
        b_lp = create_butterworth_filter(IMG_SHAPE, CUTOFF_LP, n, 'lp')
        b_hp = create_butterworth_filter(IMG_SHAPE, CUTOFF_HP, n, 'hp')
        
        axes[i, 0].imshow(img_to_compare, cmap='gray')
        axes[i, 0].set_title(f"Original (for n={n})", fontsize=12)
        
        filtered_lp = apply_frequency_filter(img_to_compare, b_lp)
        axes[i, 1].imshow(filtered_lp, cmap='gray')
        axes[i, 1].set_title(f"Butterworth LPF (n={n})", fontsize=12)
        
        filtered_hp = apply_frequency_filter(img_to_compare, b_hp)
        axes[i, 2].imshow(filtered_hp, cmap='gray')
        axes[i, 2].set_title(f"Butterworth HPF (n={n})", fontsize=12)

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("figure_4_butterworth_n_effect.png")
    plt.show()