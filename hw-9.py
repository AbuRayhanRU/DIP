import numpy as np
import cv2
import matplotlib.pyplot as plt


def show_images_and_hist(images, titles):
    """Display images and their histograms side by side."""
    plt.figure(figsize=(20, 20))
    n = len(images)
    for i, (img, title) in enumerate(zip(images, titles)):
        # Show image
        plt.subplot(2, n, i + 1)
        plt.imshow(img, cmap="gray")
        plt.title(title)
        plt.axis("off")

        # Show histogram
        plt.subplot(2, n, i + 1 + n)
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        plt.plot(hist, color="black")
        plt.xlim([0, 256])
        plt.title(f"Histogram - {title}")

    plt.tight_layout()
  #  plt.savefig("images/output/histogram_matching_variation.png")
    plt.show()


def match_hist_cdf(src, ref):
    """Histogram matching using CDF and interpolation."""
    hist_src = cv2.calcHist([src], [0], None, [256], [0, 256]).ravel()
    hist_ref = cv2.calcHist([ref], [0], None, [256], [0, 256]).ravel()

    cdf_src = hist_src.cumsum() / hist_src.sum()
    cdf_ref = hist_ref.cumsum() / hist_ref.sum()

    # Use np.interp instead of loop
    mapping = np.interp(cdf_src, cdf_ref, np.arange(256))
    lut = np.round(mapping).astype(np.uint8)

    return cv2.LUT(src, lut)


def match_hist_spec(src, ref):
    """Histogram specification using step mapping."""
    hist_src, _ = np.histogram(src.flatten(), 256, [0, 256])
    hist_ref, _ = np.histogram(ref.flatten(), 256, [0, 256])

    hist_src = hist_src.astype(np.float32) / hist_src.sum()
    hist_ref = hist_ref.astype(np.float32) / hist_ref.sum()

    cdf_src = np.cumsum(hist_src)
    cdf_ref = np.cumsum(hist_ref)

    lut = np.zeros(256, dtype=np.uint8)
    j = 0
    for i in range(256):
        while j < 255 and cdf_ref[j] < cdf_src[i]:
            j += 1
        lut[i] = j

    return cv2.LUT(src, lut)


def change_contrast(img, mode="normal"):
    """Simulate different contrast levels."""
    img = img.astype(np.float32)
    if mode == "low":
        img = img * 0.5 + 64
    elif mode == "high":
        img = (img - 128) * 2 + 128
    return np.clip(img, 0, 255).astype(np.uint8)


def histogram_correlation(im1, im2):
    """Return correlation coefficient between histograms."""
    h1 = cv2.calcHist([im1], [0], None, [256], [0, 256]).ravel()
    h2 = cv2.calcHist([im2], [0], None, [256], [0, 256]).ravel()
    return np.corrcoef(h1, h2)[0, 1]


# ----------------- MAIN PIPELINE ----------------- #
image = cv2.imread("/home/rayhan/Downloads/image7.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError("Image not found: images/deer.jpg")

# Generate test images
src_low = change_contrast(image, "low")
ref_high = change_contrast(image, "high")

# Apply histogram matching
res_cdf = match_hist_cdf(src_low, ref_high)
res_spec = match_hist_spec(src_low, ref_high)

# Display results
show_images_and_hist(
    [src_low, ref_high, res_cdf, res_spec],
    ["Source (Low)", "Reference (High)", "Matched (CDF)", "Matched (Spec)"]
)

# Print correlation
print(f"Histogram Correlation - CDF:  {histogram_correlation(ref_high, res_cdf):.4f}")
print(f"Histogram Correlation - Spec: {histogram_correlation(ref_high, res_spec):.4f}")
