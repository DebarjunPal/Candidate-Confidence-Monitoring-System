# Candidate-Confidence-Monitoring-System

## Overview

The **Candidate-Confidence-Monitoring-System** is a real-time computer vision application designed to analyze and monitor a candidate's confidence level based on facial emotions and other visual cues. Leveraging state-of-the-art deep learning libraries, this system captures video from a webcam, detects the user's face, predicts their emotional state, and computes a dynamic "confidence score." The results are visualized through a live dashboard, making it useful for mock interviews, public speaking training, online assessments, or any scenario where confidence monitoring is valuable.

---

## Features

- **Real-Time Face Detection**: Uses OpenCV to detect faces in webcam video streams.
- **Emotion Recognition**: Employs DeepFace for robust emotion analysis (e.g., happy, sad, angry, neutral, surprise, etc.).
- **Confidence Scoring**: Calculates a confidence score based on:
  - Detected emotion type (mapped to base confidence values)
  - Certainty of emotional prediction
  - Face position and size (centered, well-lit faces score higher)
  - Emotional stability over a moving window
- **Live Dashboard**: Interactive dashboard with:
  - Video feed overlay including bounding box and annotations
  - Real-time confidence score bar and history trend plot
  - Breakdown of contributing factors and primary emotion
- **User Feedback & Guidance**: Provides actionable feedback if no face or emotion is detected.
- **Performance Metrics**: Displays FPS and system status indicators.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/DebarjunPal/Candidate-Confidence-Monitoring-System.git
cd Candidate-Confidence-Monitoring-System
```

### 2. Install Dependencies

Ensure Python 3.7+ is installed. You may wish to use a virtual environment.

```bash
pip install -r requirements.txt
```

**Key dependencies:**
- `opencv-python`
- `numpy`
- `matplotlib`
- `deepface`

If `requirements.txt` is missing, install manually:

```bash
pip install opencv-python numpy matplotlib deepface
```

### 3. Ensure Camera Access

The application requires access to a webcam (default: device 0).

---

## Usage

Run the main script:

```bash
python confidence.py
```

- The app will open a window titled **"Emotion Confidence Monitor"**.
- The dashboard shows:
  - Live video with face/annotation overlays
  - Real-time confidence score and emotion
  - A trend plot of your confidence over time
  - Contributing factors and improvement suggestions

**To exit:** Press the `q` key in the dashboard window.

---

## How It Works

1. **Face Detection**: Locates the largest visible face in the frame.
2. **Emotion Analysis**: Analyzes the face ROI with DeepFace to predict emotion and its confidence.
3. **Confidence Calculation**: Combines the following:
    - Emotion base value (e.g., happy = high, sad = low)
    - Prediction certainty
    - Face size/position (centered, large faces are better)
    - Emotional stability (consistent emotion = higher confidence)
4. **Dashboard Rendering**: Composes a numpy/cv2 dashboard with:
    - Annotated video
    - Confidence score bar and plot
    - Emotion bar chart
    - Guidance text

---

## Customization

- **Emotion Weights**: You can adjust the `emotion_confidence_scores` in `confidence.py` to tune how each emotion affects the confidence score.
- **Window Size**: Dashboard and plot sizes can be changed in the class initialization.

---

## Troubleshooting

- **No Face Detected**: Ensure your face is clearly visible, well-lit, and the camera is unobstructed.
- **Camera Not Found**: Check that your webcam is plugged in and not being used by another application.
- **Dependency Errors**: Install missing packages as shown above.

---

## Contributing

Contributions are welcome! To contribute:

1. Fork this repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes and commit (`git commit -am 'Add feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

Please follow PEP8 style guidelines and include clear commit messages.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [OpenCV](https://opencv.org/)
- [DeepFace](https://github.com/serengil/deepface)
- [Matplotlib](https://matplotlib.org/)

---

## Contact

For questions or issues, please open an [issue](https://github.com/DebarjunPal/Candidate-Confidence-Monitoring-System/issues) or contact [Debarjun Pal](mailto:debarjunpal134@gmail.com).
