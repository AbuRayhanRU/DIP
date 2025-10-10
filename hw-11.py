import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

IMG_PATH = '/home/rayhan/Downloads/image7.jpg'
GRID_COUNT = 3
CLAHE_TILE_SIZE = 8



def hist_eq(image):
    """Perform global histogram equalization."""
    return cv.equalizeHist(image)


def clahe_enhance(image, clip_val=2.0, tile_dim=(8, 8)):
    """Perform CLAHE with custom clip limit and grid size."""
    clahe = cv.createCLAHE(clipLimit=clip_val, tileGridSize=tile_dim)
    return clahe.apply(image)


def ahe_with_interp(image, grid=(8, 8)):
    """Perform AHE with strong local adaptation (using high clip)."""
    return clahe_enhance(image, clip_val=100.0, tile_dim=grid)


def ahe_no_interp(image, tiles=(8, 8)):
    """Perform naive AHE without interpolation (local HE per tile)."""
    rows, cols = tiles
    h, w = image.shape
    y_splits = np.linspace(0, h, rows + 1, dtype=int)
    x_splits = np.linspace(0, w, cols + 1, dtype=int)

    output = np.empty_like(image)
    for i in range(rows):
        for j in range(cols):
            y0, y1 = y_splits[i], y_splits[i + 1]
            x0, x1 = x_splits[j], x_splits[j + 1]
            output[y0:y1, x0:x1] = cv.equalizeHist(image[y0:y1, x0:x1])
    return output


def adjust_contrast_brightness(image, alpha=1.2, beta=10):
    """Linear transformation for contrast and brightness."""
    return cv.convertScaleAbs(image, alpha=alpha, beta=beta)


def gamma_adj(image, gamma=1.0):
    """Apply gamma correction."""
    gamma = max(1e-6, float(gamma))
    inv_gamma = 1.0 / gamma
    lut = ((np.arange(256) / 255.0) ** inv_gamma * 255).clip(0, 255).astype(np.uint8)
    return cv.LUT(image, lut)


def gaussian_smooth(image, kernel=5, sigma=1.0):
    """Apply Gaussian blur for smoothing."""
    kernel = kernel if kernel % 2 == 1 else kernel + 1
    return cv.GaussianBlur(image, (kernel, kernel), sigma)


def median_smooth(image, kernel=5):
    """Apply median blur for noise removal."""
    kernel = kernel if kernel % 2 == 1 else kernel + 1
    return cv.medianBlur(image, kernel)


def sharpen_image(image, intensity=1.0, sigma=1.0):
    """Apply unsharp masking for sharpening effect."""
    blur = cv.GaussianBlur(image, (0, 0), sigma)
    return cv.addWeighted(image, 1 + intensity, blur, -intensity, 0)



def op_identity(tile):
    return tile

def op_linear_boost(tile):
    return adjust_contrast_brightness(tile, 1.2, 10)

def op_linear_reduce(tile):
    return adjust_contrast_brightness(tile, 0.9, -10)

def op_gamma07(tile):
    return gamma_adj(tile, 0.7)

def op_gamma15(tile):
    return gamma_adj(tile, 1.5)

def op_gaussian(tile):
    return gaussian_smooth(tile, 5, 1.0)

def op_median(tile):
    return median_smooth(tile, 5)

def op_sharpen(tile):
    return sharpen_image(tile, 1.0, 1.0)

def op_hist_eq(tile):
    return hist_eq(tile)

def op_ahe_interp(tile):
    return ahe_with_interp(tile, (CLAHE_TILE_SIZE, CLAHE_TILE_SIZE))

def op_clahe2(tile):
    return clahe_enhance(tile, 2.0, (CLAHE_TILE_SIZE, CLAHE_TILE_SIZE))

def op_clahe4(tile):
    return clahe_enhance(tile, 4.0, (CLAHE_TILE_SIZE, CLAHE_TILE_SIZE))


ALL_OPS = [
    op_identity,
    op_linear_boost,
    op_linear_reduce,
    op_gamma07,
    op_gamma15,
    op_gaussian,
    op_median,
    op_sharpen,
    op_hist_eq,
    op_ahe_interp,
    op_clahe2,
    op_clahe4,
]



def grid_mosaic(image, s, operations):
    """Split image into s×s grids and apply operations per grid."""
    h, w = image.shape
    y_steps = np.linspace(0, h, s + 1, dtype=int)
    x_steps = np.linspace(0, w, s + 1, dtype=int)

    result = np.zeros_like(image)
    op_index = 0
    for i in range(s):
        for j in range(s):
            y0, y1 = y_steps[i], y_steps[i + 1]
            x0, x1 = x_steps[j], x_steps[j + 1]
            patch = image[y0:y1, x0:x1]
            try:
                result[y0:y1, x0:x1] = operations[op_index % len(operations)](patch)
            except Exception:
                result[y0:y1, x0:x1] = patch
            op_index += 1
    return result



img_gray = cv.imread(IMG_PATH, cv.IMREAD_GRAYSCALE)

orig_img = img_gray
he_img = hist_eq(img_gray)
ahe_interp_img = ahe_with_interp(img_gray, (CLAHE_TILE_SIZE, CLAHE_TILE_SIZE))
clahe_2_img = clahe_enhance(img_gray, 2.0, (CLAHE_TILE_SIZE, CLAHE_TILE_SIZE))
clahe_4_img = clahe_enhance(img_gray, 4.0, (CLAHE_TILE_SIZE, CLAHE_TILE_SIZE))
ahe_no_interp_img = ahe_no_interp(img_gray, (CLAHE_TILE_SIZE, CLAHE_TILE_SIZE))
mosaic_img = grid_mosaic(img_gray, GRID_COUNT, ALL_OPS)



fig, ax = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Local Enhancement Comparison", fontsize=16)

ax[0, 0].imshow(orig_img, cmap="gray")
ax[0, 0].set_title("Original")
ax[0, 0].axis("off")

ax[0, 1].imshow(he_img, cmap="gray")
ax[0, 1].set_title("Histogram Equalization")
ax[0, 1].axis("off")

ax[0, 2].imshow(ahe_interp_img, cmap="gray")
ax[0, 2].set_title("AHE (Interpolated)")
ax[0, 2].axis("off")

ax[0, 3].imshow(clahe_2_img, cmap="gray")
ax[0, 3].set_title("CLAHE (Clip=2.0)")
ax[0, 3].axis("off")

ax[1, 0].imshow(clahe_4_img, cmap="gray")
ax[1, 0].set_title("CLAHE (Clip=4.0)")
ax[1, 0].axis("off")

ax[1, 1].imshow(ahe_no_interp_img, cmap="gray")
ax[1, 1].set_title("AHE (No Interpolation)")
ax[1, 1].axis("off")

ax[1, 2].imshow(mosaic_img, cmap="gray")
ax[1, 2].set_title("Mosaic (All Techniques)")
ax[1, 2].axis("off")

ax[1, 3].remove()

plt.tight_layout()
#plt.savefig("images/output/enhancement_comparison.png")
plt.show()
