import os, sys, argparse, yaml
import numpy as np
from typing import Optional, Dict
from skimage import exposure, img_as_ubyte, io as skio
from skimage.restoration import rolling_ball
from skimage.measure import regionprops_table
from tifffile import imread as tifread, imwrite as tifsave


# ---------- IO ----------
def read_tiff_to_ZCYX(path: str) -> np.ndarray:
    """
    Read a multi-dimensional TIFF and return (Z, C, Y, X) float32.
    Handles common axis orders like ZCYX, CZYX, ZYX, YXC, YX.
    Tries to use tifffile's axes metadata; falls back to heuristics.
    """
    import tifffile as tiff

    with tiff.TiffFile(path) as tf:
        # Prefer first series
        s = tf.series[0]
        axes = getattr(s, "axes", "") or ""
        arr = s.asarray()  # lazy-reads the series with correct shape
    arr = np.asarray(arr)

    # If axes are known, map to (Z, C, Y, X)
    # We only care about Z, C, Y, X; ignore others (e.g., T) by squeezing/merging
    axes = axes.upper()
    if axes:
        # Build a list of dims that exist
        dims = list(axes)
        data = arr
        # Ensure Y and X are last two dims (if present)
        if "Y" in dims and "X" in dims:
            y_idx, x_idx = dims.index("Y"), dims.index("X")
            target_order = [d for d in dims if d not in ("Y", "X")] + ["Y", "X"]
            perm = [dims.index(d) for d in target_order]
            data = np.moveaxis(data, range(data.ndim), perm)
            dims = target_order
        else:
            # Assume last 2 are Y,X
            pass

        # Insert missing Z and/or C dims as singleton axes at front
        if "Z" not in dims:
            data = np.expand_dims(data, 0)
            dims = ["Z"] + dims
        if "C" not in dims:
            data = np.expand_dims(data, 1)
            dims = [dims[0]] + ["C"] + dims[1:]

        # Now move to (Z,C,*,Y,X) then squeeze extras in the middle if any
        z_idx, c_idx = dims.index("Z"), dims.index("C")
        # bring Z->0, C->1
        order = [z_idx, c_idx] + [i for i, d in enumerate(dims) if d not in ("Z", "C")]
        data = np.moveaxis(data, range(data.ndim), order)

        # ensure Y,X are last two
        if data.ndim < 4:
            # pad if somehow too small
            if data.ndim == 3:
                data = data[:, :, np.newaxis, :]
                if data.shape[-1] > 1:
                    # best-effort; real-world TIFFs we use above should not hit here
                    pass

        # If we still have extra dims between C and Y, squeeze them (take first)
        while data.ndim > 4:
            data = data.take(indices=0, axis=2)

        out = data
    else:
        # No axes metadata → heuristics
        a = arr
        if a.ndim == 4:
            # Common cases: (Z,C,Y,X) or (C,Z,Y,X). Heuristic: if first dim <=5, it's probably C.
            if a.shape[0] <= 5:
                a = np.moveaxis(a, 0, 1)  # (C,Z,Y,X) -> (Z,C,Y,X)
            # else assume already (Z,C,Y,X)
            out = a
        elif a.ndim == 3:
            # Either (Z,Y,X) or (Y,X,C)
            if a.shape[-1] in (3, 4):          # (Y,X,C)
                a = np.moveaxis(a, -1, 0)[np.newaxis, ...]  # -> (1,C,Y,X)
                out = np.moveaxis(a, 0, 1)                  # -> (1,C,Y,X) already; keep Z=1
            else:                                # (Z,Y,X)
                out = a[:, np.newaxis, ...]      # -> (Z,1,Y,X)
        elif a.ndim == 2:
            out = a[np.newaxis, np.newaxis, ...]  # -> (1,1,Y,X)
        else:
            # fallback
            out = np.asarray(a)[np.newaxis, np.newaxis, ...]

    return out.astype(np.float32)

def read_tiff_to_ZCYX(path: str) -> np.ndarray:
    a = tifread(path); a = np.asarray(a)
    if a.ndim == 4:
        if a.shape[0] <= 5:  # (C,Z,Y,X) -> (Z,C,Y,X)
            a = np.moveaxis(a, 0, 1)
    elif a.ndim == 3:
        if a.shape[-1] in (3,4):   # (Y,X,C) -> (1,C,Y,X)
            a = np.moveaxis(a, -1, 0)[np.newaxis, ...]
        else:                       # (Z,Y,X) -> (Z,1,Y,X)
            a = a[:, np.newaxis, ...]
    else:
        a = a[np.newaxis, np.newaxis, ...]
    return a.astype(np.float32)

def infer_channel_indices(C: int) -> Dict[str,int]:
    return {"DAPI": 0 if C>=1 else None, "Alexa488": 1 if C>=2 else None, "Cy3": 2 if C>=3 else None}

# ---------- Fiji ----------
def init_imagej(fiji_app_dir: Optional[str]):
    import imagej
    import scyjava

    # --- scyjava options: support both new and old API ---
    add_opt = getattr(scyjava, "add_option", None)
    if callable(add_opt):
        scyjava.add_option("-Dimagej.legacy.enabled=true")
    else:
        # old API path
        try:
            from scyjava import config as sjcfg
            sjcfg.add_option("-Dimagej.legacy.enabled=true")
        except Exception as e:
            print("Warning: could not set scyjava option via either API:", e)

    # --- init ImageJ from local Fiji (recommended on Windows) ---
    if fiji_app_dir:
        return imagej.init(fiji_app_dir, mode="headless", add_legacy=True)
    else:
        # fallback to Maven (requires JDK & network)
        return imagej.init("sc.fiji:fiji", mode="headless", add_legacy=True)


def fiji_max_intensity(ij, np_stack_ZYX: np.ndarray) -> np.ndarray:
    import scyjava
    ZProjector = scyjava.jimport("ij.plugin.ZProjector")
    img = ij.py.to_img(np_stack_ZYX.astype(np.float32))
    imp = ij.py.to_imageplus(img)
    zp = ZProjector(imp)
    zp.setMethod(ZProjector.MAX_METHOD)
    zp.setStartSlice(1); zp.setStopSlice(imp.getNSlices()); zp.doProjection()
    mip_imp = zp.getProjection()
    mip = ij.py.from_java(mip_imp)
    mip = np.asarray(mip, dtype=np.float32)
    vmin, vmax = float(mip.min()), float(mip.max())
    return np.zeros_like(mip, dtype=np.float32) if vmax<=vmin+1e-12 else (mip-vmin)/(vmax-vmin)

# ---------- Processing ----------
def process_channel(stack_ZYX: np.ndarray, z_index, rolling_radius=50, ij=None) -> np.ndarray:
    Z = stack_ZYX.shape[0]
    if z_index == "middle":
        img = stack_ZYX[Z//2]
    elif isinstance(z_index, int):
        zi = max(0, min(Z-1, z_index)); img = stack_ZYX[zi]
    else:
        img = fiji_max_intensity(ij, stack_ZYX) if ij is not None else stack_ZYX.max(axis=0)
    bg = rolling_ball(img, radius=rolling_radius)
    img_corr = img - bg; img_corr[img_corr<0] = 0
    return exposure.rescale_intensity(img_corr, out_range=(0,1)).astype(np.float32)

def save_previews(dapi, alexa, cy3, prefix):
    if dapi is not None:
        skio.imsave(f"{prefix}_DAPI_blue.jpg", img_as_ubyte(np.dstack([np.zeros_like(dapi), np.zeros_like(dapi), dapi])), check_contrast=False)
    if alexa is not None:
        skio.imsave(f"{prefix}_Alexa488_green.jpg", img_as_ubyte(np.dstack([np.zeros_like(alexa), alexa, np.zeros_like(alexa)])), check_contrast=False)
    if cy3 is not None:
        skio.imsave(f"{prefix}_Cy3_red.jpg", img_as_ubyte(np.dstack([cy3, np.zeros_like(cy3), np.zeros_like(cy3)])), check_contrast=False)
    ref = next(x for x in [dapi, alexa, cy3] if x is not None)
    H,W = ref.shape; merge = np.zeros((H,W,3), dtype=np.float32)
    if cy3   is not None: merge[...,0]=cy3
    if alexa is not None: merge[...,1]=alexa
    if dapi  is not None: merge[...,2]=dapi
    skio.imsave(f"{prefix}_RGBmerge.jpg", img_as_ubyte(merge), check_contrast=False)

# ---------- StarDist ----------
def _normalize_percentile(img: np.ndarray, lo=2.0, hi=98.5) -> np.ndarray:
    lo, hi = np.percentile(img, [lo, hi]).astype(np.float32)
    if hi<=lo+1e-12: return np.zeros_like(img, dtype=np.float32)
    x = (img-lo)/(hi-lo); x[x<0]=0; x[x>1]=1; return x.astype(np.float32)

def run_stardist_2d(dapi_img01: np.ndarray, model_name="2D_versatile_fluo",
                    prob=0.579071, nms=0.30, pct=(2.0,98.5)) -> np.ndarray:
    from stardist.models import StarDist2D
    x = _normalize_percentile(dapi_img01, *pct)
    model = StarDist2D.from_pretrained(model_name)
    labels, details = model.predict_instances(x, prob_thresh=float(prob), nms_thresh=float(nms))
    return labels.astype(np.uint16)

def measure_labels(labels: np.ndarray):
    import pandas as pd
    props = regionprops_table(labels, properties=("label","area","centroid","bbox"))
    return pd.DataFrame(props)

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Fiji MIP + StarDist + measurement pipeline")
    ap.add_argument("--config", type=str, required=True, help="YAML config path")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    inp = cfg["input_path"]
    prefix = cfg["save_prefix"]
    os.makedirs(os.path.dirname(prefix), exist_ok=True)

    mode = cfg.get("mode","fiji_mip")
    chmap = cfg.get("channels",{}).get("mapping", {"DAPI":0,"Alexa488":1,"Cy3":2})
    stc = cfg.get("stardist", {})
    sd_run = stc.get("run", True)
    sd_channel = stc.get("channel","DAPI")
    sd_idx = chmap.get(sd_channel,0) if isinstance(sd_channel,str) else int(sd_channel)
    fiji_dir = cfg.get("fiji",{}).get("app_dir", None)

    # read
    if inp.lower().endswith((".oib",".oif")):
        stack = read_oib_oif_to_ZCYX(inp)
    else:
        stack = read_tiff_to_ZCYX(inp)

    Z,C,Y,X = stack.shape
    dapi_c, alexa_c, cy3_c = chmap.get("DAPI",0), chmap.get("Alexa488",1), chmap.get("Cy3",2)

    # Fiji if needed
    ij = init_imagej(fiji_dir) if mode=="fiji_mip" else None

    # per-channel processing
    dapi  = process_channel(stack[:, dapi_c],  z_index=None if mode!="middle" else "middle", ij=ij) if dapi_c is not None and dapi_c<C else None
    alexa = process_channel(stack[:, alexa_c], z_index=None if mode!="middle" else "middle", ij=ij) if alexa_c is not None and alexa_c<C else None
    cy3   = process_channel(stack[:, cy3_c],   z_index=None if mode!="middle" else "middle", ij=ij) if cy3_c is not None and cy3_c<C else None

    # previews
    save_previews(dapi, alexa, cy3, prefix)

    # StarDist
    if sd_run:
        mip_for_sd = [dapi, alexa, cy3][sd_idx]
        if mip_for_sd is None:
            raise ValueError(f"No image for StarDist channel index {sd_idx}")
        labels = run_stardist_2d(mip_for_sd,
                                 model_name=stc.get("model","2D_versatile_fluo"),
                                 prob=stc.get("prob_thresh",0.579071),
                                 nms=stc.get("nms_thresh",0.30),
                                 pct=tuple(stc.get("normalize_percentiles",[2.0,98.5])))
        lab_path = f"{prefix}_stardist_labels.tif"
        tifsave(lab_path, labels, dtype=np.uint16)
        df = measure_labels(labels)
        csv_path = f"{prefix}_stardist_results.csv"
        df.to_csv(csv_path, index=False)
        print("Saved:", lab_path, csv_path)

    print("Done.")

if __name__ == "__main__":
    main()
