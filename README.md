# 🎧 Speech Processing and Transcription using OpenAI Whisper

This project uses OpenAI's Whisper model for automatic speech recognition (ASR), combined with `librosa` for audio processing and `scikit-learn` for optional data analysis or machine learning tasks.

## 🛠️ Requirements

Before running the project, make sure you have Python 3.8+ and the following packages installed:

```bash
pip install openai-whisper librosa scikit-learn
Additional CUDA Requirements (for GPU acceleration)
If you're using GPU, ensure you have the correct NVIDIA CUDA drivers installed as Whisper requires PyTorch with CUDA support.

📁 Project Structure

.
├── audio_samples/           # Folder to store input audio files
├── scripts/
│   ├── transcribe.py        # Script for audio transcription using Whisper
│   ├── analyze_audio.py     # Optional: audio analysis using librosa
├── requirements.txt         # List of required packages
└── README.md                # You're here!
🚀 How to Use
1. Transcribe Audio
You can transcribe an audio file (WAV, MP3, etc.) using:


python scripts/transcribe.py --audio_path audio_samples/sample.wav
2. Analyze Audio (Optional)
If you want to extract audio features using librosa:


python scripts/analyze_audio.py --audio_path audio_samples/sample.wav
📦 Sample Code
transcribe.py

import whisper
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--audio_path", required=True, help="Path to audio file")
args = parser.parse_args()

model = whisper.load_model("base")
result = model.transcribe(args.audio_path)
print("Transcription:", result["text"])
analyze_audio.py

import librosa
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--audio_path", required=True, help="Path to audio file")
args = parser.parse_args()

y, sr = librosa.load(args.audio_path)
tempo, _ = librosa.beat.beat_track(y, sr=sr)
print(f"Estimated tempo: {tempo:.2f} BPM")
🔬 Features
🎤 Speech-to-text using OpenAI Whisper

🎼 Audio feature extraction using Librosa

🧠 Machine learning ready with Scikit-learn

⚡ CUDA-enabled for faster inference

📌 Notes
Supported audio formats: WAV, MP3, FLAC, etc.

Transcriptions may vary in accuracy depending on background noise, accents, and audio quality.

📜 License
This project is open-source under the MIT License.


# Facial Emotion Detection with Mini-XCEPTION

This project detects human facial emotions (like Happy, Sad, Angry, etc.) from uploaded images using a **pre-trained Mini-XCEPTION model** and OpenCV's Haar Cascade face detector. It is implemented in **Google Colab** and designed to be beginner-friendly.

---

## 📂 Features

- ✅ Detect faces from uploaded images using OpenCV
- ✅ Preprocess face region (64x64 grayscale)
- ✅ Predict emotions using the pre-trained Mini-XCEPTION model
- ✅ Draw bounding boxes and emotion labels on the image
- ✅ Display the result within Google Colab

---

## 🔧 Dependencies

Install the required libraries:

```python
!pip install -q keras opencv-python
📥 Download Pre-trained Emotion Model

!wget -q https://github.com/oarriaga/face_classification/raw/master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5 -O emotion_model.h5
🧠 Supported Emotion Labels

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
📸 How It Works
Upload an image using files.upload() in Google Colab.

Convert the image to grayscale.

Detect faces using Haar Cascade.

Resize the face to 64x64 and normalize it.

Predict emotion using the Mini-XCEPTION model.

Display the image with labeled emotion.

🚀 Sample Execution

from google.colab import files
uploaded = files.upload()

img = cv2.imread("your_image.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

for (x, y, w, h) in faces:
    roi = gray[y:y+h, x:x+w]
    roi = cv2.resize(roi, (64, 64))
    roi = roi.astype("float32") / 255.0
    roi = np.expand_dims(roi, axis=-1)
    roi = np.expand_dims(roi, axis=0)

    preds = model.predict(roi, verbose=0)
    label = emotion_labels[np.argmax(preds)]

    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

cv2_imshow(img)
📌 Notes
Only grayscale 64x64 images are supported by the model.

Best results are achieved with clear frontal face images.

Use cv2_imshow() instead of cv2.imshow() in Google Colab.

📚 References
Mini-XCEPTION Model (by oarriaga)

FER-2013 Dataset

🙋‍♀️ Developed by
Jushna Breethi, Computer Science Engineering Student
🚀 Passionate about AI, Machine Learning, and Robotics

