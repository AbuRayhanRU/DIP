import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def adjust_contrast(image, alpha, beta):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def plot_on_ax(ax, data, title, cmap="gray"):
    ax.imshow(data, cmap=cmap)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def resize_image(image, target_size=(256, 256)):
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


image_paths = [
    "images/image.jpg",
    "images/image4.jpg",
    "images/image2.jpg",
    "images/image3.jpg",
    "images/image7.jpg",
]

num_images = 5
cols_per_row = 6
num_rows = num_images * 2  
num_cols = cols_per_row

fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 2.5))

os.makedirs("images/output", exist_ok=True)

col_titles = [
    "Original",
    "Low Contrast",
    "High Contrast",
    "Normal LPF",
    "Normal HPF",
    "Normal BPF",
    "Low LPF",
    "Low HPF",
    "Low BPF",
    "High LPF",
    "High HPF",
    "High BPF",
]

for img_idx in range(num_images):
    image_path = image_paths[img_idx]

    
    row1 = img_idx * 2
    row2 = img_idx * 2 + 1


    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
  
    original_image = resize_image(original_image)


    
    axs[row1, 0].set_ylabel(
        os.path.basename(image_path), rotation=90, size="large", labelpad=20
    )

    low_contrast_image = adjust_contrast(original_image, alpha=0.5, beta=50)
    high_contrast_image = adjust_contrast(original_image, alpha=2.0, beta=-100)

    images = {
        "Normal": original_image,
        "Low": low_contrast_image,
        "High": high_contrast_image,
    }

    
    plot_on_ax(axs[row1, 0], images["Normal"], col_titles[0])
    plot_on_ax(axs[row1, 1], images["Low"], col_titles[1])
    plot_on_ax(axs[row1, 2], images["High"], col_titles[2])

    rows, cols = original_image.shape
    crow, ccol = rows // 2, cols // 2

    mask_lpf = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask_lpf, (ccol, crow), 30, 1, -1)
    mask_hpf = np.ones((rows, cols), np.uint8)
    cv2.circle(mask_hpf, (ccol, crow), 30, 0, -1)
    mask_bpf = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask_bpf, (ccol, crow), 80, 1, -1)
    cv2.circle(mask_bpf, (ccol, crow), 20, 0, -1)

    filters = {"LPF": mask_lpf, "HPF": mask_hpf, "BPF": mask_bpf}

    f_transforms_shifted = {}
    contrast_types = ["Normal", "Low", "High"]

    for j, contrast in enumerate(contrast_types):
        img = images[contrast]
        f_transform = np.fft.fft2(img)
        f_transforms_shifted[contrast] = np.fft.fftshift(f_transform)

    col_idx = 3  

    for k, (filter_name, mask) in enumerate(filters.items()):
        f_filtered_shifted = f_transforms_shifted["Normal"] * mask
        f_inverse_shifted = np.fft.ifftshift(f_filtered_shifted)
        img_filtered = np.abs(np.fft.ifft2(f_inverse_shifted))
        img_filtered_normalized = cv2.normalize(
            img_filtered, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

        plot_on_ax(axs[row1, col_idx], img_filtered_normalized, col_titles[col_idx])
        col_idx += 1

    col_idx = 0
    
    for k, (filter_name, mask) in enumerate(filters.items()):
        f_filtered_shifted = f_transforms_shifted["Low"] * mask
        f_inverse_shifted = np.fft.ifftshift(f_filtered_shifted)
        img_filtered = np.abs(np.fft.ifft2(f_inverse_shifted))
        img_filtered_normalized = cv2.normalize(
            img_filtered, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

        plot_on_ax(axs[row2, col_idx], img_filtered_normalized, col_titles[6 + col_idx])
        col_idx += 1

    
    for k, (filter_name, mask) in enumerate(filters.items()):
        f_filtered_shifted = f_transforms_shifted["High"] * mask
        f_inverse_shifted = np.fft.ifftshift(f_filtered_shifted)
        img_filtered = np.abs(np.fft.ifft2(f_inverse_shifted))
        img_filtered_normalized = cv2.normalize(
            img_filtered, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

        plot_on_ax(axs[row2, col_idx], img_filtered_normalized, col_titles[6 + col_idx])
        col_idx += 1

    print(f"Processed row for {image_path}")

for j in range(6):
    if j < len(col_titles):
        axs[0, j].set_title(col_titles[j], fontsize=10, pad=20)


row2_titles = ["Low LPF", "Low HPF", "Low BPF", "High LPF", "High HPF", "High BPF"]
for j in range(6):
    if j < len(row2_titles):
        axs[1, j].set_title(row2_titles[j], fontsize=10, pad=20)


for i in range(2, num_rows):
    for j in range(num_cols):
        axs[i, j].set_title("")

plt.tight_layout(rect=[0.02, 0.0, 1, 0.98])

output_filename = "images/output/all_in_one_figure.png"

# plt.savefig(output_filename, dpi=200, bbox_inches="tight")
# plt.close()
plt.show()

print(f"\nProcessing complete. All results saved in '{output_filename}'")