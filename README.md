# Candidate-Confidence-Monitoring-System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.7%2B-blue?logo=python&logoColor=white&style=for-the-badge" alt="Python"/>
  <img src="https://img.shields.io/badge/OpenCV-Enabled-brightgreen?logo=opencv&logoColor=white&style=for-the-badge" alt="OpenCV"/>
  <img src="https://img.shields.io/badge/DeepFace-Emotion%20Recognition-yellow?style=for-the-badge" alt="DeepFace"/>
  <img src="https://img.shields.io/badge/Matplotlib-Plotting-orange?logo=matplotlib&logoColor=white&style=for-the-badge" alt="Matplotlib"/>
---

## 🚀 Overview

**Candidate-Confidence-Monitoring-System** is a real-time computer vision application designed to analyze and monitor a candidate's confidence level based on facial emotions and other visual cues.Leveraging state-of-the-art deep learning libraries, this system captures video from a webcam, detects the user's face, predicts their emotional state, and computes a dynamic "confidence score." The results are visualized through a live dashboard, making it useful for mock interviews, public speaking training, online assessments, or any scenario where confidence monitoring is valuable.


---

## ✨ Features

- 🕵️ **Real-Time Face Detection** (OpenCV)
- 😃 **Emotion Recognition** (DeepFace)
- 📈 **Confidence Scoring** and Trend Visualization (Matplotlib)
- 🖥️ **Live Dashboard**: Video feed overlay, confidence score, emotion breakdown
- 💡 **User Feedback & Guidance** if no face or emotion is detected
- ⚡ **Performance Metrics**: FPS and system status indicators

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/DebarjunPal/Candidate-Confidence-Monitoring-System.git
cd Candidate-Confidence-Monitoring-System
```

### 2. Install Dependencies

Make sure Python 3.7+ is installed. Recommended: use a virtual environment.

```bash
pip install -r requirements.txt
```
**Key dependencies:**  
`opencv-python`, `numpy`, `matplotlib`, `deepface`

---

## ▶️ Usage

Run the main script:

```bash
python confidence.py
```

- The dashboard window displays:
  - Live video with face/annotation overlays
  - Real-time confidence score and emotion
  - Trend plot
  - Suggestions and contributing factors

**To exit:** Press `q` in the dashboard window.

---

## 🧠 How It Works

1. **Face Detection:** Locates the largest visible face in the frame.
2. **Emotion Analysis:** DeepFace predicts emotion and confidence.
3. **Confidence Calculation:** Combines emotion value, certainty, face position/size, and emotional stability.
4. **Dashboard Rendering:** Shows annotated video, confidence score, emotion chart, and improvement guidance.

---

## 🧩 Customization

- **Tune Emotion Weights:** Adjust `emotion_confidence_scores` in `confidence.py`
- **Change Dashboard Size:** Modify dashboard/plot size in initialization

---

## 🆘 Troubleshooting

- ❌ **No Face Detected:** Ensure your face is visible, well-lit, and camera is unobstructed
- 📷 **Camera Not Found:** Check webcam connection and usage status
- 🛠️ **Dependency Errors:** Install missing packages as shown above

---

## 🤝 Contributing

1. Fork this repo
2. Create a feature branch
3. Make changes, commit, and push
4. Open a Pull Request

Follow PEP8 and write clear commit messages.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [OpenCV](https://opencv.org/)
- [DeepFace](https://github.com/serengil/deepface)
- [Matplotlib](https://matplotlib.org/)

---

## 📬 Contact

Open an [issue](https://github.com/DebarjunPal/Candidate-Confidence-Monitoring-System/issues) or email [Debarjun Pal](mailto:debarjunpal134@gmail.com).
