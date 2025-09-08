import cv2
import numpy as np
import matplotlib.pyplot as plt

# ================= Histogram Equalization (Custom) =================
def custom_histogram_equalization(gray_img):
    """Perform histogram equalization on a grayscale image manually."""

    # Step 1: Compute Histogram
    histogram = np.bincount(gray_img.flatten(), minlength=256)

    # Step 2: Compute PDF (Probability Distribution Function)
    pdf = histogram / gray_img.size

    # Step 3: Compute CDF (Cumulative Distribution Function)
    cdf = np.cumsum(pdf)

    # Step 4: Normalize and map to new intensity values
    transform_map = np.floor(255 * cdf).astype(np.uint8)

    # Step 5: Apply transformation
    equalized_img = transform_map[gray_img]

    return equalized_img, histogram, pdf, cdf


# ================= Visualization Function =================
def plot_comparison(original, custom_eq, opencv_eq, hist_orig, hist_custom, hist_opencv):
    """Display images and their histograms side by side."""

    fig, axes = plt.subplots(3, 2, figsize=(12, 15))

    # Row 1: Original image and histogram
    axes[0, 0].imshow(original, cmap="gray")
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].bar(range(256), hist_orig, color="gray")
    axes[0, 1].set_title("Original Histogram")

    # Row 2: Custom Equalized
    axes[1, 0].imshow(custom_eq, cmap="gray")
    axes[1, 0].set_title("Custom Equalized Image")
    axes[1, 0].axis("off")

    axes[1, 1].bar(range(256), hist_custom, color="black")
    axes[1, 1].set_title("Histogram (Custom)")

    # Row 3: OpenCV Equalized
    axes[2, 0].imshow(opencv_eq, cmap="gray")
    axes[2, 0].set_title("OpenCV Equalized Image")
    axes[2, 0].axis("off")

    axes[2, 1].bar(range(256), hist_opencv, color="black")
    axes[2, 1].set_title("Histogram (OpenCV)")

    plt.tight_layout()
    plt.show()


# ================= Main =================
def main():
    # Load image in grayscale
    img_path = '/home/rayhan/Downloads/image7.jpg'
    gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if gray_img is None:
        print("Error: Image not found at the given path.")
        return

    # Apply custom histogram equalization
    custom_eq_img, hist_orig, pdf, cdf = custom_histogram_equalization(gray_img)

    # Apply OpenCV histogram equalization
    opencv_eq_img = cv2.equalizeHist(gray_img)

    # Compute histograms for equalized images
    hist_custom = np.bincount(custom_eq_img.flatten(), minlength=256)
    hist_opencv = np.bincount(opencv_eq_img.flatten(), minlength=256)

    # Display results
    plot_comparison(gray_img, custom_eq_img, opencv_eq_img,
                    hist_orig, hist_custom, hist_opencv)


if __name__ == "__main__":
    main()
