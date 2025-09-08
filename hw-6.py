import cv2
import numpy as np
import matplotlib.pyplot as plt

#==================== Main Function ========================================
def main():
    # Load grayscale image
    img_gray = cv2.imread('/home/rayhan/Downloads/image7.jpg', cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError("Image not found. Check your path!")

    # Add Gaussian noise
    row, col = img_gray.shape
    mean, var = 0, 0.01
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col))
    noisy_img = img_gray + gauss * 255
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

    # ================== Define Kernels ==================
    avg_filter = np.ones((3, 3), np.float32) / 9

    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)

    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=np.float32)

    prewitt_x = np.array([[-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]], dtype=np.float32)

    prewitt_y = np.array([[-1, -1, -1],
                          [ 0,  0,  0],
                          [ 1,  1,  1]], dtype=np.float32)

    laplace_4 = np.array([[ 0, -1,  0],
                          [-1,  4, -1],
                          [ 0, -1,  0]], dtype=np.float32)

    scharr_x = np.array([[-3, 0, 3],
                         [-10, 0, 10],
                         [-3, 0, 3]], dtype=np.float32)

    scharr_y = np.array([[-3, -10, -3],
                         [ 0,   0,  0],
                         [ 3,  10,  3]], dtype=np.float32)

    # Four custom kernels
    emboss = np.array([[-2, -1, 0],
                       [-1,  1, 1],
                       [ 0,  1, 2]], dtype=np.float32)

    motion_blur = np.ones((1, 5), np.float32) / 5

    unsharp = np.array([[ 0, -1,  0],
                        [-1,  5, -1],
                        [ 0, -1,  0]], dtype=np.float32)

    diagonal_enhance = np.array([[-1, -1, 0],
                                 [-1,  0, 1],
                                 [ 0,  1, 1]], dtype=np.float32)

    filters = {
        "Avg": avg_filter,
        "Sobel-X": sobel_x,
        "Sobel-Y": sobel_y,
        "Prewitt-X": prewitt_x,
        "Prewitt-Y": prewitt_y,
        "Laplace-4": laplace_4,
        "Scharr-X": scharr_x,
        "Scharr-Y": scharr_y,
        "Emboss": emboss,
        "MotionBlur": motion_blur,
        "Unsharp": unsharp,
        "DiagonalEnhance": diagonal_enhance
    }

    # Display setup
    img_set = [img_gray, noisy_img]
    img_title = ['Original', 'Gaussian Noise']

    # Apply filters
    for name, kernel in filters.items():
        f_same = filter_same(noisy_img, kernel)
        f_valid = filter_valid(noisy_img, kernel)

        img_set.extend([f_same, f_valid])
        img_title.extend([f'{name} (same)', f'{name} (valid)'])

    # Show results
    display(img_set, img_title)


#==================== Manual Convolution: VALID (no padding) ===============
def filter_valid(input_img, kernel):
    tmp_img = input_img.astype(np.float32)
    input_h, input_w = input_img.shape
    kernel_h, kernel_w = kernel.shape
    output_h = input_h - kernel_h + 1
    output_w = input_w - kernel_w + 1

    output_img = np.zeros((output_h, output_w), dtype=np.float32)
    for h in range(output_h):
        for w in range(output_w):
            roi = tmp_img[h:h+kernel_h, w:w+kernel_w]
            output_img[h, w] = np.sum(roi * kernel)

    return np.clip(output_img, 0, 255).astype(np.uint8)


#==================== Manual Convolution: SAME (with padding) ===============
def filter_same(input_img, kernel):
    tmp_img = input_img.astype(np.float32)
    kernel_h, kernel_w = kernel.shape
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2

    padded_img = np.pad(tmp_img, ((pad_h, pad_h), (pad_w, pad_w)),
                        mode='constant', constant_values=0)
    output_h, output_w = input_img.shape
    output_img = np.zeros((output_h, output_w), dtype=np.float32)

    for h in range(output_h):
        for w in range(output_w):
            roi = padded_img[h:h+kernel_h, w:w+kernel_w]
            output_img[h, w] = np.sum(roi * kernel)

    return np.clip(output_img, 0, 255).astype(np.uint8)


#==================== Display Function (Dynamic Layout) ====================
def display(img_set, img_title, cols=4):
    num_images = len(img_set)
    rows = (num_images + cols - 1) // cols

    plt.figure(figsize=(5 * cols, 5 * rows))
    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img_set[i], cmap='gray')
        plt.title(img_title[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()


#==================== Run Script ====================
if __name__ == "__main__":
    main()
