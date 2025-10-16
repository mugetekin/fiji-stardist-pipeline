import argparse, numpy as np
import tifffile as tiff
from csbdeep.utils import normalize
from stardist.models import StarDist2D

def load_channel_mip(path, channel=0, mode="mip"):
    arr = tiff.imread(path)
    # Expect (Z, C, H, W) like (7,3,1024,1024). Fallbacks for common shapes.
    if arr.ndim == 4:
        Z, C, H, W = arr.shape
        if C <= channel:
            raise ValueError(f"Channel index {channel} out of range for shape {arr.shape}")
        stack = arr[:, channel]  # (Z,H,W)
    elif arr.ndim == 3:
        # If (Z,H,W), assume single channel
        stack = arr
    else:
        raise ValueError(f"Unsupported TIFF shape {arr.shape}")

    if mode == "mip":
        img = stack.max(axis=0).astype(np.float32)
    elif mode == "middle":
        img = stack[stack.shape[0]//2].astype(np.float32)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Percentile normalization (2,98.5) like your config
    p2, p985 = np.percentile(img, (2.0, 98.5))
    if p985 > p2:
        img = (img - p2) / (p985 - p2)
    img = np.clip(img, 0, 1)
    return img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input .tif")
    ap.add_argument("--output", required=True, help="Full path for <prefix>_stardist_labels.tif")
    ap.add_argument("--channel", type=int, default=0, help="DAPI channel index (default 0)")
    ap.add_argument("--mode", default="mip", choices=["mip","middle"])
    ap.add_argument("--prob_thresh", type=float, default=0.58)
    ap.add_argument("--nms_thresh", type=float, default=0.30)
    args = ap.parse_args()

    img = load_channel_mip(args.input, channel=args.channel, mode=args.mode)

    # Load pretrained StarDist model (downloads weights on first use)
    model = StarDist2D.from_pretrained("2D_versatile_fluo")
    labels, _ = model.predict_instances(img, prob_thresh=args.prob_thresh, nms_thresh=args.nms_thresh)

    # Save as uint16 label image
    tiff.imwrite(args.output, labels.astype(np.uint16))
    print(f"[saved] {args.output}")

if __name__ == "__main__":
    main()
