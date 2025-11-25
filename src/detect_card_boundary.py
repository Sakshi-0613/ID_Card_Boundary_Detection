import os
import cv2
import numpy as np
from ultralytics import YOLO
import argparse


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect.astype(int)


def get_model():
    # project_root = one level above src
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    model_path = os.path.join(project_root, "models", "best.pt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    print(f"\n[INFO] Loading model from: {model_path}")
    return YOLO(model_path)


def get_best_box(results):
    """Return highest-confidence bbox as (x1,y1,x2,y2) ints, or None."""
    boxes = results.boxes
    if len(boxes) == 0:
        return None

    confs = boxes.conf.cpu().numpy()
    idx = int(np.argmax(confs))
    x1, y1, x2, y2 = boxes[idx].xyxy[0].cpu().numpy()
    return int(x1), int(y1), int(x2), int(y2)


def expand_box(x1, y1, x2, y2, W, H, margin=0.25):
    """Expand box by margin fraction, keep inside image."""
    bw, bh = x2 - x1, y2 - y1
    ex1 = max(0, int(x1 - bw * margin))
    ey1 = max(0, int(y1 - bh * margin))
    ex2 = min(W - 1, int(x2 + bw * margin))
    ey2 = min(H - 1, int(y2 + bh * margin))
    return ex1, ey1, ex2, ey2


# ---- main pipeline ----

def detect_card_boundary(image_path: str, output_dir: str = None):
    if not os.path.exists(image_path):
        print(f"\n!!!!!!Image not found: {image_path}")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"\n!!!!!!Failed to read image: {image_path}")
        return

    H, W = img.shape[:2]

    # 1) YOLO detection
    model = get_model()
    results = model.predict(img, conf=0.25, verbose=False)[0]

    bbox = get_best_box(results)
    if bbox is None:
        print("\n!!!!!!No id_card detected by YOLO.")
        return

    x1, y1, x2, y2 = bbox
    print(f"[INFO] YOLO box: {x1}, {y1}, {x2}, {y2}")

    # 2) Expand YOLO box so full card is guaranteed inside
    ex1, ey1, ex2, ey2 = expand_box(x1, y1, x2, y2, W, H, margin=0.25)
    crop = img[ey1:ey2, ex1:ex2].copy()
    ch, cw = crop.shape[:2]

    if ch < 20 or cw < 20:
        print("\n!!!!!!Crop too small, cannot refine.")
        return

    # 3) Downscale crop for faster GrabCut
    target_w = 500
    scale = target_w / float(cw) if cw > target_w else 1.0
    small_w = int(cw * scale)
    small_h = int(ch * scale)
    crop_small = cv2.resize(crop, (small_w, small_h), interpolation=cv2.INTER_LINEAR)

    # 4) Prepare GrabCut mask and rect (using YOLO box relative to crop)
    #    Transform original YOLO box to crop-local coords, then to small.
    local_x1 = max(0, x1 - ex1)
    local_y1 = max(0, y1 - ey1)
    local_x2 = min(cw - 1, x2 - ex1)
    local_y2 = min(ch - 1, y2 - ey1)

    gx1 = int(local_x1 * scale)
    gy1 = int(local_y1 * scale)
    gx2 = int(local_x2 * scale)
    gy2 = int(local_y2 * scale)

    g_w = max(1, gx2 - gx1)
    g_h = max(1, gy2 - gy1)
    grab_rect = (gx1, gy1, g_w, g_h)

    mask = np.zeros(crop_small.shape[:2], np.uint8)
    bgModel = np.zeros((1, 65), np.float64)
    fgModel = np.zeros((1, 65), np.float64)

    # 5) Run GrabCut
    cv2.grabCut(crop_small, mask, grab_rect, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)

    mask_fg = np.where((mask == 1) | (mask == 3), 255, 0).astype("uint8")

    # 6) Upscale mask back to crop size
    mask_full = cv2.resize(mask_fg, (cw, ch), interpolation=cv2.INTER_NEAREST)

    # 7) Edge detection & contours on mask
    edges = cv2.Canny(mask_full, 50, 150)
    edges = cv2.dilate(edges, None, iterations=2)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        print("\n!!!!No contour found on mask, falling back to YOLO rectangle.")
        quad = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    else:
        # largest contour
        c = max(cnts, key=cv2.contourArea)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) >= 4:
            if len(approx) > 4:
                # Use minAreaRect for clean quad
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                quad_local = box.astype(int)
            else:
                quad_local = approx.reshape(4, 2)
        else:
            # fall back to minAreaRect
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            quad_local = box.astype(int)

        quad_local = order_points(quad_local)

        # map from crop coords back to full image
        quad = quad_local.copy()
        quad[:, 0] += ex1
        quad[:, 1] += ey1

    # 8) Draw final boundary on original image
    out = img.copy()
    cv2.polylines(out, [quad.reshape(-1, 1, 2)], True, (0, 255, 255), 3)

    # 9) Save to boundary_output
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    if output_dir is None:
        output_dir = os.path.join(project_root, "boundary_output")

    os.makedirs(output_dir, exist_ok=True)

    base = os.path.basename(image_path)
    name, ext = os.path.splitext(base)
    out_path = os.path.join(output_dir, f"{name}_boundary{ext}")

    cv2.imwrite(out_path, out)
    print(f"\n✅ Saved boundary output: {out_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ID card boundary detector (YOLO + GrabCut + edges)")
    parser.add_argument("--image", help="Path to a single image")
    parser.add_argument("--folder", help="Folder containing multiple images")
    args = parser.parse_args()

    # --- Single Image Mode ---
    if args.image:
        detect_card_boundary(args.image)

    # --- Batch Folder Mode ---
    elif args.folder:
        folder = args.folder

        if not os.path.exists(folder):
            print(f"\n!!!!!Folder not found: {folder}")
            exit()

        supported_ext = (".jpg", ".jpeg", ".png", ".bmp")

        print("\n📂 Batch mode enabled")
        print(f"→ Reading images from: {folder}")

        for filename in os.listdir(folder):
            if filename.lower().endswith(supported_ext):

                img_path = os.path.join(folder, filename)
                print(f"\n=== Processing: {img_path} ===")
                detect_card_boundary(img_path)

        print("\n✅ Batch processing completed!")

    else:
        print("\n!!!Please provide either --image or --folder")