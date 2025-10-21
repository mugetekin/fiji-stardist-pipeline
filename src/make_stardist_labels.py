"""
StarDist Segmentation Script
----------------------------
This script loads a microscopy TIFF stack, extracts the DAPI (or other nuclear) channel,
applies either a Maximum Intensity Projection (MIP) or middle-slice extraction,
normalizes the image, and then segments nuclei using a pretrained StarDist 2D model.

Usage example:
--------------
python run_stardist_segmentation.py \
    --input data/sample.tif \
    --output outputs/sample_stardist_labels.tif \
    --channel 0 \
    --mode mip
"""

import argparse, numpy as np
import tifffile as tiff
from csbdeep.utils import normalize
from stardist.models import StarDist2D

# Helper function: load a specific fluorescence channel and
# apply projection or slicing to obtain a 2D image.
def load_channel_mip(path, channel=0, mode="mip"):
    # Load the image stack. Typical shape is (Z, C, H, W)
    arr = tiff.imread(path)
    # Expect (Z, C, H, W) like (7,3,1024,1024). Fallbacks for common shapes.
    if arr.ndim == 4:  # Handle various TIFF shapes gracefully
        Z, C, H, W = arr.shape # Expecting 4D stack: (Z, C, H, W)
        if C <= channel:
            raise ValueError(f"Channel index {channel} out of range for shape {arr.shape}")
        stack = arr[:, channel]  # (Z,H,W)   # Extract the desired channel → (Z, H, W)
    elif arr.ndim == 3:
        # If the image is already 3D (Z, H, W), assume single channel
        stack = arr
    else:
        raise ValueError(f"Unsupported TIFF shape {arr.shape}")

    # Convert Z-stack to 2D image using the selected mode
    if mode == "mip":
        # Maximum intensity projection (brightest pixel along Z)
        img = stack.max(axis=0).astype(np.float32)
    elif mode == "middle":
        # Take the middle slice along Z
        img = stack[stack.shape[0]//2].astype(np.float32)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Normalize intensity for model compatibility.
    # Using robust percentile normalization (2–98.5%)
    # to reduce the effect of outliers and background.
    p2, p985 = np.percentile(img, (2.0, 98.5))
    if p985 > p2:
        img = (img - p2) / (p985 - p2)
    img = np.clip(img, 0, 1)
    return img

# Main function: parse arguments, run StarDist prediction, save output.
def main():
    # Parse command-line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input .tif")
    ap.add_argument("--output", required=True, help="Full path for <prefix>_stardist_labels.tif")
    ap.add_argument("--channel", type=int, default=0, help="DAPI channel index (default 0)")
    ap.add_argument("--mode", default="mip", choices=["mip","middle"])
    ap.add_argument("--prob_thresh", type=float, default=0.58)
    ap.add_argument("--nms_thresh", type=float, default=0.30)
    args = ap.parse_args()

    # Load and preprocess the image
    img = load_channel_mip(args.input, channel=args.channel, mode=args.mode)

    # Load a pretrained 2D StarDist model.
    # This will download the "2D_versatile_fluo" weights on first use
    # (trained on many fluorescence microscopy datasets).
    model = StarDist2D.from_pretrained("2D_versatile_fluo")
    # Perform instance segmentation (nucleus detection)
    # Outputs:
    #   - labels : 2D label mask (each cell = unique integer)
    #   - details : auxiliary info (e.g., polygons, probabilities)
    labels, _ = model.predict_instances(img, prob_thresh=args.prob_thresh, nms_thresh=args.nms_thresh)

    # Save segmentation mask as uint16 TIFF
    # Each pixel corresponds to a labeled nucleus region.
    tiff.imwrite(args.output, labels.astype(np.uint16))
    print(f"[saved] {args.output}")

if __name__ == "__main__":
    main()
