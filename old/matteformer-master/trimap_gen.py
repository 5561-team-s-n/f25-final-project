import os
import cv2
import numpy as np
import argparse

def gen_trimap(alpha, max_kernel_size=30):
    """
    Generate a trimap using MatteFormer's erosion method.
    alpha: uint8 grayscale alpha matte (0–255)
    """

    alpha = alpha.astype(np.float32) / 255.0

    # Foreground mask = alpha = 1
    fg_mask = (alpha + 1e-5).astype(int).astype(np.uint8)
    # Background mask = alpha = 0
    bg_mask = (1 - alpha + 1e-5).astype(int).astype(np.uint8)

    # Prebuild kernels like MatteFormer
    kernels = [
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        for ks in range(1, max_kernel_size + 1)
    ]

    # Randomly choose kernel sizes (same as MatteFormer training)
    fg_erode_kernel = kernels[np.random.randint(1, max_kernel_size)]
    bg_erode_kernel = kernels[np.random.randint(1, max_kernel_size)]

    fg_mask_eroded = cv2.erode(fg_mask, fg_erode_kernel)
    bg_mask_eroded = cv2.erode(bg_mask, bg_erode_kernel)

    # Create the trimap: 128 = unknown
    trimap = np.ones_like(alpha, dtype=np.uint8) * 128
    trimap[fg_mask_eroded == 1] = 255
    trimap[bg_mask_eroded == 1] = 0

    return trimap


def process_folder(alpha_dir, trimap_dir, max_kernel_size=30):
    os.makedirs(trimap_dir, exist_ok=True)

    for filename in os.listdir(alpha_dir):
        in_path = os.path.join(alpha_dir, filename)
        out_path = os.path.join(trimap_dir, filename)

        alpha = cv2.imread(in_path, 0)
        if alpha is None:
            print(f"Skipping {filename} (not an image)")
            continue

        trimap = gen_trimap(alpha, max_kernel_size=max_kernel_size)

        cv2.imwrite(out_path, trimap)
        print(f"Generated trimap for {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha-dir", type=str, required=True,
                        help="Folder containing alpha mattes")
    parser.add_argument("--trimap-dir", type=str, required=True,
                        help="Folder to save generated trimaps")
    parser.add_argument("--max-kernel-size", type=int, default=30,
                        help="Max erosion kernel size (MatteFormer default ≈ 30)")

    args = parser.parse_args()

    process_folder(args.alpha_dir, args.trimap_dir, args.max_kernel_size)
