# 🎨 AI Virtual Canvas using Hand Gestures

An AI-powered virtual drawing canvas that lets you draw on the screen
using just your fingers, powered by computer vision.

## 🚀 Features
- Draw using index finger
- Pinch (index + middle finger) to erase
- Real-time hand tracking
- Smooth drawing experience

## 🧠 How it Works
- Uses MediaPipe to detect hand landmarks
- Tracks finger positions via OpenCV
- Maps gestures to draw / erase actions

## 🛠 Tech Stack
- Python
- OpenCV
- MediaPipe
- NumPy

## ▶️ How to Run
```bash
pip install -r requirements.txt
python main.py
