# Esma'a
Esma’a is a mobile application designed to bridge the communication gap between Deaf/Hard-of-Hearing individuals and hearing people. The app recognizes Arabic Sign Language (ArSL) hand gestures using advanced Computer Vision and Deep Learning, then converts them into clear Arabic text and spoken speech.
This system aims to make communication more accessible, inclusive, and instant for users.

## Features
- **Real-Time Sign Recognition:** Detects Arabic Sign Language gestures live using MediaPipe and a fine-tuned CNN model (EfficientNetB0).
- **Sign-to-Text Conversion:** Translates recognized gestures into Arabic characters and full sentences.
- **Text-to-Speech Output:** Uses Microsoft Edge-TTS to convert translated text into clear Arabic audio.
- **Cross-Platform Mobile App:** Built using Flutter for Android and iOS.

## Technologies Used
- **Flutter:** Mobile app development
- **FastAPI:** Backend server for recognition & TTS
- **EfficientNetB0:** Deep learning model for classification
- **MediaPipe:** hand tracking
- **OpenCV:** Image preprocessing
- **Microsoft Edge TTS:** Arabic text-to-speech
- **PyTorch / TorchVision:** Training and modeling

## System Workflow
- **Hand Detection:** MediaPipe detects the hand and extracts 21 landmarks.
- **Preprocessing:** Frames are resized, normalized, and cropped.
- **Gesture Classification:** EfficientNetB0 predicts the Arabic letter.
- **Sentence Building:** Users can add, delete, or clear letters to form sentences.
- **Speech Generation:** The final text is converted to Arabic audio using Edge-TTS.

## Screen Structure

### 1. Welcome Screen:
- Entry/intro and navigation to about us (من نحن) or try it now (!جرب الآن).

<br>

<img src="results/welcom_page.PNG" alt="App Screenshot" width="200"/>

### 2. About us Screen:
- Team members and project purpose.

<br>

<img src="results/about_us.PNG" alt="App Screenshot" width="200"/>


### 3. Home Screen:
- main page where you can navigate to dictionary and camera.

<br>

<img src="results/home.PNG" alt="App Screenshot" width="200"/>

### 4. dictionary Screen:
- Displays all supported signs with search tap.

<br>

<img src="results/dictionary.PNG" alt="App Screenshot" width="200"/>


### 5. Camera Screen:
- Live camera feed and Real-time ArSL prediction with info icon.

<br>

<p>
  <img src="results/camera_info.PNG" alt="App Screenshot" width="200"/>
  <img src="results/camera.PNG" alt="App Screenshot" width="200"/>
</p>


---

## Back-End Structure

### 1. Preprocessing Script (Dataset Preparation)

A Python script (e.g., `preprocess_dataset.py`) that:

- Reads raw dataset images (e.g., ASLAD-190K) from `SRC_ROOT`.
- Uses **MediaPipe Hands** to:
  - Detect the hand.
  - Compute a square bounding box with configurable padding.
- Crops the hand area, pads to a square, resizes to `224×224`, and saves to `DST_ROOT`.
- Logs failures (no hand, invalid crop, etc.) into a JSON file for inspection.

### 2. Training Script (Model Training)

A Python script (e.g., `train_efficientnet.py`) that:

- Loads processed train/val/test splits via `torchvision.datasets.ImageFolder`.
- Applies:
  - Data augmentation (flips, rotations, affine transforms, color jitter).
  - Normalization with ImageNet mean and standard deviation.
- Fine-tunes **EfficientNet-B0**:
  - Replaces the final classifier layer with `Num_class = 33`.
  - Uses **Adam** optimizer, **CrossEntropyLoss**, and **ReduceLROnPlateau** scheduler.
  - Early stopping when validation accuracy stops improving.
- Evaluates on the test set:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1 Score**
  - **Average inference delay per image**
- Plots and displays a confusion matrix.
- Saves the best model weights (e.g., `Final_best_efficientnet_ASLAD.pth`).

### 3. Server Script (`server.py`)

- Loads the trained EfficientNet-B0 model and runs it in **evaluation mode**.
- Uses **MediaPipe Hands** at runtime to detect and crop the hand from each incoming frame.
- Preprocessing pipeline for each frame:
  - Crop → pad to square → resize to `224×224` → normalization.
- Predicts the class index and maps it to the corresponding Arabic letter via `arabic_map`.


## ✅ Requirements

Before running this project, make sure you have the following installed on your system:

### 🔧 Backend Requirements
- :contentReference[oaicite:0]{index=0} 3.8 or higher  
- :contentReference[oaicite:1]{index=1}  
- :contentReference[oaicite:2]{index=2}  
- :contentReference[oaicite:3]{index=3}  
- :contentReference[oaicite:4]{index=4}  
- :contentReference[oaicite:5]{index=5}  
- :contentReference[oaicite:6]{index=6}  
- :contentReference[oaicite:7]{index=7}  
- :contentReference[oaicite:8]{index=8}  

---

### 📱 Frontend Requirements
- :contentReference[oaicite:9]{index=9} (latest stable version)
- :contentReference[oaicite:10]{index=10}
- :contentReference[oaicite:11]{index=11} or any supported IDE
- Android Emulator or a physical Android device

---

### 🗂 General Tools
- :contentReference[oaicite:12]{index=12}  
- A stable internet connection (for package installation)


## Demo Video
[Watch the Demo Video](https://drive.google.com/file/d/10Ya4PQ8kg312SVhIkqgdBpU0FUgdg09P/view?usp=sharing)


## Future Plan:
- Support for full words & sentences
- Integration of facial expression recognition
- Offline speech synthesis
- Multi-language support

## References:
- [ ASLAD190K dataset](https://www.kaggle.com/datasets/salmaselmoghazy/aslad-190k-arabic-sign-language-alphabet-dataset)
- [ ASL Alphabet datase](https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset) 
  

## Team member:
- Wateen Ali Alrumayh
- Renad Majed Alrubaish
- Rahaf Raied Megdad
- Dina Hameed Alotaibi

