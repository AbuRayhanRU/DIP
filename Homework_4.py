import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('/home/rayhan/Downloads/image7.jpg', cv2.IMREAD_GRAYSCALE)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_float = img.astype(np.float32)

threshold_values = [80, 128, 180]
thresholded_images = []
for t in threshold_values:
    _, th_img = cv2.threshold(img, t, 255, cv2.THRESH_BINARY)
    thresholded_images.append((f"Threshold = {t}", th_img))

c = 255 / np.log(1 + np.max(img_float))
log_transformed = c * (np.log(img_float + 1))
log_transformed = np.array(log_transformed, dtype=np.uint8)

gamma = 2.2
gamma_corrected = np.array(255 * (img_float / 255.0) ** gamma, dtype=np.uint8)

fig, axes = plt.subplots(2, 4, figsize=(16, 9))
axes = axes.ravel()

axes[0].imshow(img, cmap='gray')
axes[0].set_title("Original Image")
axes[0].axis("off")

for i, (title, th_img) in enumerate(thresholded_images, start=1):
    axes[i].imshow(th_img, cmap='gray')
    axes[i].set_title(title)
    axes[i].axis("off")

axes[4].imshow(log_transformed, cmap='gray')
axes[4].set_title("Log Transformation")
axes[4].axis("off")

axes[5].hist(log_transformed.ravel(), bins=256, range=(0, 256), color='black')
axes[5].set_title("Log Histogram")

axes[6].imshow(gamma_corrected, cmap='gray')
axes[6].set_title("Gamma Correction")
axes[6].axis("off")

axes[7].hist(gamma_corrected.ravel(), bins=256, range=(0, 256), color='black')
axes[7].set_title("Gamma Histogram")

plt.tight_layout()
# plt.show()
plt.savefig('thresholding_transformation.png')