# server.py
# -- coding: utf-8 --
import os, json, base64, atexit, tempfile 
from collections import deque

import cv2, torch, torch.nn as nn
from torchvision import models, transforms
import numpy as np

# MediaPipe for hand detection
import mediapipe as mp

# FastAPI + WebSocket
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Edge TTS
import edge_tts


# :::::::::::::: Variables definition :::::::::::
MARGIN        = 40 # Extra space added around the detected hand box
USE_SMOOTH    = True # Enable smoothing of predictions across multiple frames
SMOOTH_N      = 4 # Number of frames used for smoothing the output prediction
CONF_TH       = 0.90 # Minimum confidence required to accept the model's prediction
IMG_SIZE      = 224 # Model input size (width and height) for model input
USE_MEDIAPIPE = True # Use MediaPipe for hand detection

# :::::::::::::: Classes ordering ::::::::::::::
class_names = [
    "ain","al","aleff","bb","dal","dha","dhad","fa","gaaf","ghain",
    "ha","haa","jeem","kaaf","khaa","la","laam","meem","nun","ra",
    "saad","seen","sheen","space","ta","taa","thaa","thal","toot",
    "waw","ya","yaa","zay"
] # List of all classes the model can predict
idx_to_class = {i: n for i, n in enumerate(class_names)}  # Create dict: index → class name

# :::::::::::::: Arabic Leters map ::::::::::::
arabic_map = {
    "ain":"ع","al":"ال","aleff":"ا","bb":"ب","dal":"د","dha":"ظ","dhad":"ض","fa":"ف",
    "gaaf":"ق","ghain":"غ","ha":"ه","haa":"ح","jeem":"ج","kaaf":"ك","khaa":"خ","la":"لا",
    "laam":"ل","meem":"م","nun":"ن","ra":"ر","saad":"ص","seen":"س","sheen":"ش","space":" ",
    "ta":"ط","taa":"ت","thaa":"ث","thal":"ذ","toot":"ة","waw":"و","ya":"ي","yaa":"ئ","zay":"ز"
} # Map model labels to Arabic letters

# :::::::::::::: Transform settings ::::::::::::::
transform = transforms.Compose([
    transforms.ToPILImage(),     # Convert the image from OpenCV format (NumPy) to PIL format
    transforms.Resize((IMG_SIZE, IMG_SIZE)), # Resize the image to 224x224 because EfficientNet needs this size
    transforms.ToTensor(), # Convert the image to a Tensor and scale pixel values from 0–255 to 0–1
    transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                         std =(0.229, 0.224, 0.225)) # Normalize the image using ImageNet mean and std
])

# :::::::::::::: Pad to square ::::::::::::::
# Make the cropped image a square by adding black padding if needed
def pad_to_square(img_bgr):
    h, w = img_bgr.shape[:2] # Get the size of the image after cropping ( the size of hight and width)
    if h == w: # if image after copping is square 
        return img_bgr # dont do anything, return the image
    s = max(h, w)  # the larger edge : hight or width
    # Compute top and bottom padding
    top  = (s - h) // 2 
    bot  = s - h - top
    # Compute left and right padding
    left = (s - w) // 2
    rgt  = s - w - left
    # Add black border to make the image square
    return cv2.copyMakeBorder(img_bgr, top, bot, left, rgt,
                              cv2.BORDER_CONSTANT, value=(0, 0, 0)) # Add black borders to make the image square by using function from OpenCV

# ::::::::::::::: Model ::::::::::::::
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Choose GPU if available, otherwise use CPU
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1) #same as imagNet weights
in_f = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_f, len(class_names)) # Replace the last layer with our number of classes 
state = torch.load(r"Last_best_efficientnet_ASLAD.pth", map_location=device)
model.load_state_dict(state, strict=True) 
model.to(device).eval()  # set the model to evaluation mode

# :::::::::::::: Mediapipe ::::::::::::::
mp_hands = mp.solutions.hands # Load the MediaPipe Hands solution
hands = None
if USE_MEDIAPIPE:
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

@atexit.register #when the system close, close mediapipe also
def _cleanup():
    try:
        if hands is not None:
            hands.close()
    except:
        pass

# =============== inference function ===============
@torch.no_grad() # in inference phase
def infer_one(bgr_frame):
    H, W = bgr_frame.shape[:2] # Get frame siz
    x_min, y_min, x_max, y_max = 0, 0, W, H
    if USE_MEDIAPIPE and hands is not None:
        rgb_full = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB) # Convert BGR to RGB

        res = hands.process(rgb_full)

        if res.multi_hand_landmarks: # if hand is detected
            hand_lm = res.multi_hand_landmarks[0] #contains hand landmarks 
            x_min, y_min = W, H 
            x_max, y_max = 0, 0
            for lm in hand_lm.landmark:
                x, y = int(lm.x * W), int(lm.y * H)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y) #نحسب اكبر واصغر x&Y to generate bounding box
            x_min = max(0, x_min - MARGIN) #add margin to the generated box
            y_min = max(0, y_min - MARGIN)
            x_max = min(W, x_max + MARGIN)
            y_max = min(H, y_max + MARGIN)
            if x_max <= x_min or y_max <= y_min: #if values not accepted 
                return "", 0.0
        else: #if no hand detected 
            return "", 0.0 

    hand_bgr = bgr_frame[y_min:y_max, x_min:x_max] # Crop hand area
    if hand_bgr.size == 0:  # If empty
        return "", 0.0

    hand_bgr = pad_to_square(hand_bgr) # Make image square
    rgb = cv2.cvtColor(hand_bgr, cv2.COLOR_BGR2RGB)  
    img_tensor = transform(rgb).unsqueeze(0).to(device)
    logits = model(img_tensor)  # Get model output
    probs  = torch.softmax(logits, dim=1)[0]  # Convert output to probabilities
    conf_val, pred_idx = torch.max(probs, dim=0) # Get highest probability and its index
    conf_val = float(conf_val.item())  # Convert from tensor to float
    pred_idx = int(pred_idx.item()) # Convert from tensor to int
    pred_class = idx_to_class.get(pred_idx, None) # Convert index to class name

    if (pred_class is None) or (conf_val < CONF_TH):
        return "", conf_val
    return arabic_map.get(pred_class, ""), conf_val  # Convert class to Arabic letter


# ================= FastAPI =================
app = FastAPI() # Create API application
app.add_middleware( 
    CORSMiddleware, ## Allow cross-domain requests
    allow_origins=["*"], # Allow all origins
    allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

@app.get("/health") # Simple endpoint to test connection
async def health():
    return {"ok": True}


# ================== TTS Endpoint ==================
@app.post("/tts") # Convert text to speech
async def tts_endpoint(request: dict): 
    """
    يستقبل {"text": "مرحبا"} ويرجع ملف MP3 بصيغة base64
    """
    text = request.get("text", "").strip()
    if not text:
        return {"error": "missing text"}

    try:
        voice = "ar-SA-HamedNeural"  
        communicate = edge_tts.Communicate(text, voice=voice) #convert text to speech
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            temp_path = fp.name

        await communicate.save(temp_path)
        with open(temp_path, "rb") as f:
            audio_bytes = f.read()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        os.remove(temp_path)
        return {"audio": audio_b64}
    except Exception as e:
        return {"error": str(e)}


# ================= WebSocket =================
@app.websocket("/ws") #used to sending life frames from camera
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()  # Accept WebSocket connection
    hist = deque(maxlen=SMOOTH_N) if USE_SMOOTH else None  # Keep history for smoothing
    try:
        while True:
            raw = await websocket.receive_text() # Receive frame (base64 string)
            try:
                data = json.loads(raw)
            except Exception as e:
                await websocket.send_text(json.dumps({"error": f"invalid json: {e}"}))
                continue

            if "frame" not in data:
                await websocket.send_text(json.dumps({"error": "missing 'frame'"}))
                continue

            b64 = data["frame"]
            try:
                img_bytes = base64.b64decode(b64)# Decode base64 string
                nparr = np.frombuffer(img_bytes, np.uint8)  # Convert bytes to NumPy array
                bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Convert to OpenCV image
                if bgr is None:
                    await websocket.send_text(json.dumps({"error": "decode failed"}))
                    continue
            except Exception as e:
                await websocket.send_text(json.dumps({"error": f"decode error: {e}"}))
                continue

            char, conf = infer_one(bgr) # Run prediction

            if USE_SMOOTH:
                hist.append((char, conf)) #يحفظ n احتمالات
                if len(hist) == hist.maxlen:
                    non_empty = [c for c, s in hist if c.strip()] # Filter empty outputs
                    if non_empty:
                        best = max(set(non_empty), key=non_empty.count)  # Most common prediction
                        char = best
                        conf = max([s for c, s in hist if c == best] or [conf])

            await websocket.send_text(json.dumps({"result": char, "conf": round(conf, 4)})) #تحويل النتيجه الى json وترسل الى العميل
    except WebSocketDisconnect: # Client closed connection
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"error": f"server: {e}"}))
        except:
            pass


if __name__ == "_main_": #تشغيل السيرفر 
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)


