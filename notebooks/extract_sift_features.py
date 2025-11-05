"""
extract_sift_features.py

Module to extract SIFT features from a specific band of an image, divided into blocks.

Features:
- Optional 8-bit percentile stretch (1-99 percentile)
- Optional Sobel edge filter
- Automatic linear min-max stretch to 8-bit for SIFT
- Parallel block-wise SIFT extraction
- Optional visualization of keypoints
- Tuple-style return of block_info and optional arrays

Author: Yuvraj1729 @ Glx
"""

import cv2
import numpy as np
import rasterio
from concurrent.futures import ThreadPoolExecutor
from skimage.filters import sobel
import matplotlib.pyplot as plt

def extract_sift_features(
    image_path,
    band_number=1,
    image_row=3,
    image_col=3,
    nfeatures=130,
    nOctaveLayers=9,
    contrastThreshold=0.01,
    edgeThreshold=15,
    sigma=1.2,
    eightbit_stretch=False,
    sobel_filter=False,
    return_keypoints=False,
    return_descriptors=False,
    return_coords=False,
    display_figure=False
):
    """
    Extract SIFT features from an image band divided into blocks.

    Parameters
    ----------
    image_path : str
        File path to the image (GeoTIFF or similar).
    band_number : int
        Band index to read (1-based).
    image_row : int
        Number of rows to divide the image into.
    image_col : int
        Number of columns to divide the image into.
    nfeatures : int
        Maximum number of SIFT features per block.
    nOctaveLayers, contrastThreshold, edgeThreshold, sigma : SIFT parameters
    eightbit_stretch : bool
        Apply 1–99 percentile stretch to 8-bit before processing.
    sobel_filter : bool
        Apply Sobel edge filter before SIFT.
    return_keypoints, return_descriptors, return_coords : bool
        Flags to include extra outputs.
    display_figure : bool
        If True, plots all keypoints on the image.

    Returns
    -------
    tuple
        block_info, and optionally all_keypoints, all_descriptors, all_keypoints_coords
        depending on flags.
    """

    # -------------------------
    # Load band
    # -------------------------
    with rasterio.open(image_path) as src:
        img = src.read(band_number)
    height, width = img.shape
    #print(f"[INFO] Loaded band {band_number} from {image_path}, shape={img.shape}")
    print(f"[INFO] Loaded band {band_number} from {image_path}, shape={img.shape}, dtype={img.dtype}")


    # -------------------------
    # Sequential preprocessing
    # -------------------------
    proc_img = img.copy()  # start with original image

    # Optional 8-bit percentile stretch
    if eightbit_stretch:
        ref_min, ref_max = np.nanpercentile(proc_img, (1, 99))
        proc_img = np.clip(((proc_img - ref_min) / (ref_max - ref_min)) * 255, 0, 255)
        print("[INFO] 8-bit percentile stretch applied")

    # Optional Sobel filter
    if sobel_filter:
        proc_img = sobel(proc_img) * 255
        print("[INFO] Sobel filter applied")

    # Ensure 8-bit for SIFT
    if proc_img.dtype != np.uint8:
        img_min, img_max = np.nanmin(proc_img), np.nanmax(proc_img)
        if img_max > img_min:
            proc_img = ((proc_img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            proc_img = np.zeros_like(proc_img, dtype=np.uint8)
        print("[INFO] Linear min-max stretch applied to convert image to 8-bit for SIFT")

    # -------------------------
    # Initialize SIFT
    # -------------------------
    sift = cv2.SIFT_create(
        nfeatures=nfeatures,
        nOctaveLayers=nOctaveLayers,
        contrastThreshold=contrastThreshold,
        edgeThreshold=edgeThreshold,
        sigma=sigma,
    )
    print("[INFO] SIFT detector initialized")

    # -------------------------
    # Block division
    # -------------------------
    block_size_y = height // image_row
    block_size_x = width // image_col
    print(f"[INFO] Dividing image into {image_row} x {image_col} blocks "
          f"of approx {block_size_y} x {block_size_x} pixels")

    block_info = {}
    all_kps, all_desc, all_coords = [], [], []

    def process_block(block_number, i, j):
        x_start = j * block_size_x
        y_start = i * block_size_y
        x_end = min((j + 1) * block_size_x, width)
        y_end = min((i + 1) * block_size_y, height)

        block = proc_img[y_start:y_end, x_start:x_end]
        kps, desc = sift.detectAndCompute(block, None)

        info = {"block_number": block_number, "status": 0, "num_points": 0, "keypoints": []}

        if desc is not None:
            if len(kps) > nfeatures:
                kps = sorted(kps, key=lambda kp: kp.response, reverse=True)[:nfeatures]
                desc = desc[:nfeatures]

            coords = cv2.KeyPoint_convert(kps)
            coords[:, 0] += x_start
            coords[:, 1] += y_start

            info["status"] = 1
            info["num_points"] = len(kps)
            info["keypoints"] = [{"coord": c, "response": kp.response} for kp, c in zip(kps, coords)]

            return kps, desc, coords, info

        return [], [], [], info

    # -------------------------
    # Parallel execution
    # -------------------------
    futures = []
    with ThreadPoolExecutor() as executor:
        for i in range(image_row):
            for j in range(image_col):
                block_number = i * image_col + j
                futures.append(executor.submit(process_block, block_number, i, j))

    for f in futures:
        kps, desc, coords, info = f.result()
        all_kps.extend(kps)
        all_desc.extend(desc)
        all_coords.extend(coords)
        block_info[info["block_number"]] = info
        #print(f"[INFO] Block {info['block_number']} processed, keypoints={info['num_points']}")

    print("[INFO] Feature extraction completed")

    # -------------------------
    # Optional visualization
    # -------------------------
    if display_figure and len(all_coords) > 0:
        plt.figure(figsize=(10, 10))
        plt.imshow(proc_img, cmap='gray')
        all_coords_array = np.array(all_coords)
        plt.scatter(all_coords_array[:, 0], all_coords_array[:, 1], c='r', s=10)
        plt.title("SIFT Keypoints")
        plt.show()
        print("[INFO] Displayed keypoints on image")

    # -------------------------
    # Prepare outputs
    # -------------------------
    outputs = [block_info]
    if return_keypoints:
        outputs.append(np.array(all_kps, dtype=object))
    if return_descriptors:
        outputs.append(np.array(all_desc, dtype=object))
    if return_coords:
        outputs.append(np.array(all_coords, dtype=object))

    return tuple(outputs)


# -------------------------
# Usage examples
# -------------------------
if __name__ == "__main__":
    """
    Example 1: Only block info
    --------------------------
    block_info = extract_sift_features("image.tif", band_number=3)
    print(block_info.keys())

    Example 2: Block info + all arrays
    ----------------------------------
    block_info, kps, desc, coords = extract_sift_features(
        "image.tif",
        band_number=3,
        return_keypoints=True,
        return_descriptors=True,
        return_coords=True
    )
    print(f"Total keypoints: {len(kps)}, descriptors shape: {desc.shape}")

    Example 3: Display keypoints
    ----------------------------
    block_info, kps, desc, coords = extract_sift_features(
        "image.tif",
        band_number=3,
        return_keypoints=True,
        return_descriptors=True,
        return_coords=True,
        display_figure=True
    )
    """
    pass
