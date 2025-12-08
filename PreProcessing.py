import os, json, pathlib, cv2, numpy as np
from tqdm import tqdm #progression line
import mediapipe as mp

# ========== إعدادات ثابتة ==========
SRC_ROOT = "data/images/ASLAD-190K"      #source folder 
DST_ROOT = "ASLAD-processed_fixedALLPad1.5Res1024MinCon0.5ModelComp1"   #output folder
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

SIZE = 224               #resized size
PADDING = 1.5            
RESIZE_DET = 1024        #maximum image size forMediaPipe
MIN_CONF = 0.50          #confidence of MediaPipe
MODEL_COMPLEXITY = 1  
MAX_NUM_HANDS = 1

SUFFIX = "_crop" #added to processed image
FAIL_LOG_PATH = "metadata/fail_log_batch.json" #failed images

# ==================== Helping functions ====================

#iterates over all images in the folders
def iter_images(root_dir):
    root = pathlib.Path(root_dir)
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield str(p)

#reads images by Opencv
def read_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"cv2.imread failed for: {path}")
    return img

#make the image square --add black borders 
def pad_to_square(img_bgr):
    h, w = img_bgr.shape[:2] #extract w/h of image
    if h == 0 or w == 0:
        return None
    if h == w: 
        return img_bgr
    s = max(h, w)
    pad_top    = (s - h) // 2 
    pad_bottom = s - h - pad_top
    pad_left   = (s - w) // 2
    pad_right  = s - w - pad_left
    return cv2.copyMakeBorder( #Opencv adds black borders 
        img_bgr, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0) #black color in BGR format 
    )


#detect hand and make bounding box (x1,y1,x2,y2) with padding added to it and within image boundres 
def detect_square_bbox(img_bgr, hands):
    H, W = img_bgr.shape[:2]
    scale = 1.0
    #resize image to smaller size if needed for faster processing 
    if max(H, W) > RESIZE_DET:
        scale = RESIZE_DET / max(H, W) #compute ratio to downsize the image
        small = cv2.resize(img_bgr, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_AREA)
    else:
        small = img_bgr

    #convert image to RGB required for MediaPipe 
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb) #send image to mediapipe 
    if not res or not res.multi_hand_landmarks:
        return None

    lm = res.multi_hand_landmarks[0] #select the first detected hand
    lm_np = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32) #convert 21 marks to numpy array

    W_s, H_s = small.shape[1], small.shape[0]
    xs = (lm_np[:, 0] * W_s) / max(scale, 1e-6)
    ys = (lm_np[:, 1] * H_s) / max(scale, 1e-6)

    x1_raw, x2_raw = xs.min(), xs.max()
    y1_raw, y2_raw = ys.min(), ys.max()

    w, h = x2_raw - x1_raw, y2_raw - y1_raw #compute h/w of hand bounding box
    side = int(max(w, h) * (1 + PADDING))

    #Center of bounding box
    cx, cy = int((x1_raw + x2_raw) / 2), int((y1_raw + y2_raw) / 2) #compute the center point 
    half = side // 2
    x1 = int(np.clip(cx - half, 0, W - 1))
    x2 = int(np.clip(cx + half, 0, W - 1))
    y1 = int(np.clip(cy - half, 0, H - 1))
    y2 = int(np.clip(cy + half, 0, H - 1))

    if x2 <= x1 or y2 <= y1: #validate the bounding box has positive value
        return None
    return x1, y1, x2, y2 #return bounding box coordinates 


#process one image intirely: detect, crop, pad, resize, and save
def process_one_image(src_path, dst_path, hands, fail_log):
    try:
        img = read_image(src_path)
        bbox = detect_square_bbox(img, hands)
        if bbox is None:
            fail_log.append({"path": src_path, "reason": "no_hand_detected"})
            return False

        x1, y1, x2, y2 = bbox
        crop = img[y1:y2, x1:x2]

        if crop.size == 0:
            fail_log.append({"path": src_path, "reason": "empty_crop"})
            return False

        crop = pad_to_square(crop)
        if crop is None:
            fail_log.append({"path": src_path, "reason": "pad_failed"})
            return False

        out = cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        ok = cv2.imwrite(dst_path, out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            fail_log.append({"path": src_path, "reason": "imwrite_failed"})
            return False
        return True
    except Exception as e:
        fail_log.append({"path": src_path, "reason": f"exception:{repr(e)}"})
        return False

# ==================== main ====================
def main():
    os.makedirs(DST_ROOT, exist_ok=True) #create output folders 
    os.makedirs(os.path.dirname(FAIL_LOG_PATH), exist_ok=True) 

    classes = [d for d in os.listdir(SRC_ROOT) if os.path.isdir(os.path.join(SRC_ROOT, d))] #take all classes from sorce

    #intialize MediaPipe hand
    hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=MAX_NUM_HANDS,
        model_complexity=MODEL_COMPLEXITY,
        min_detection_confidence=MIN_CONF
    )

    fail_log = [] #list for all fail images 
    try:
        for cls in classes: 
            src_dir = os.path.join(SRC_ROOT, cls)
            dst_dir = os.path.join(DST_ROOT, cls)
            os.makedirs(dst_dir, exist_ok=True)

            files = list(iter_images(src_dir))
            if not files:
                print(f"Skipping empty class: {cls}")
                continue

            print(f"Processing class: {cls} | Found {len(files)} images")
            for p in tqdm(files, desc=f"[{cls}]", unit="img"):
                base = os.path.splitext(os.path.basename(p))[0]
                out_path = os.path.join(dst_dir, base + SUFFIX + ".jpg")
                if os.path.exists(out_path):
                    continue
                process_one_image(p, out_path, hands, fail_log)
    finally:
        try: hands.close()
        except: pass

        with open(FAIL_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(fail_log, f, ensure_ascii=False, indent=2)

        print("Done")
        print("Output:", DST_ROOT)
        print("Failed:", len(fail_log), "→", FAIL_LOG_PATH)

if __name__ == "__main__": #start main 
    main()

