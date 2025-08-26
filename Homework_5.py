import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image in grayscale
img = cv2.imread('your_image.jpg', cv2.IMREAD_GRAYSCALE)

# 1. Smoothing / Average Kernel (3x3)
kernel_avg = np.ones((3, 3), np.float32) / 9
avg_filtered = cv2.filter2D(img, -1, kernel_avg)

# 2. Sobel Kernels
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)
sobel_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]], dtype=np.float32)

sobel_x_filtered = cv2.filter2D(img, -1, sobel_x)
sobel_y_filtered = cv2.filter2D(img, -1, sobel_y)

# 3. Prewitt Kernels
prewitt_x = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]], dtype=np.float32)
prewitt_y = np.array([[-1, -1, -1],
                      [ 0,  0,  0],
                      [ 1,  1,  1]], dtype=np.float32)

prewitt_x_filtered = cv2.filter2D(img, -1, prewitt_x)
prewitt_y_filtered = cv2.filter2D(img, -1, prewitt_y)

# 4. Laplacian Kernel
laplace_kernel = np.array([[0, -1, 0],
                           [-1, 4, -1],
                           [0, -1, 0]], dtype=np.float32)
laplace_filtered = cv2.filter2D(img, -1, laplace_kernel)

# Show results using matplotlib
titles = ['Original', 'Average Blur',
          'Sobel X', 'Sobel Y',
          'Prewitt X', 'Prewitt Y',
          'Laplacian']

images = [img, avg_filtered,
          sobel_x_filtered, sobel_y_filtered,
          prewitt_x_filtered, prewitt_y_filtered,
          laplace_filtered]

plt.figure(figsize=(10, 8))
for i in range(len(images)):
    plt.subplot(3, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
